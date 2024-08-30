using KernelAbstractions.Extras.LoopInfo: @unroll


abstract type SamplerIntegrator <: Integrator end

struct WhittedIntegrator{C<: Camera, S <: AbstractSampler} <: SamplerIntegrator
    camera::C
    sampler::S
    max_depth::Int64
end

function sample_kernel_inner(i::A, scene::B, t_sampler::C, film::D, film_tile::E, camera::F, pixel::G, spp_sqr::H) where {A, B, C, D, E, F, G, H}
    for _ in 1:t_sampler.samples_per_pixel
        camera_sample = get_camera_sample(t_sampler, pixel)
        ray, ω = generate_ray_differential(camera, camera_sample)
        ray = scale_differentials(ray, spp_sqr)
        l = RGBSpectrum(0f0)
        if ω > 0.0f0
            # l = li(t_sampler, Int32(i.max_depth), ray, scene, Int32(1))
            l = li_iterative(t_sampler, Int32(i.max_depth), ray, scene)
        end
        # TODO check l for invalid values
        if isnan(l)
            l = RGBSpectrum(0f0)
        end
        add_sample!(film, film_tile, camera_sample.film, l, ω)
    end
end

@noinline function sample_kernel(i, camera, scene, film, film_tile, tile_bounds)
    t_sampler = deepcopy(i.sampler)
    spp_sqr = 1f0 / √Float32(t_sampler.samples_per_pixel)
    for pixel in tile_bounds
        sample_kernel_inner(i, scene, t_sampler, film, film_tile, camera, pixel, spp_sqr)
    end
    merge_film_tile!(film, film_tile)
end

"""
Render scene.
"""
function (i::SamplerIntegrator)(scene::Scene, film)
    sample_bounds = get_sample_bounds(film)
    sample_extent = diagonal(sample_bounds)
    tile_size = 16
    n_tiles = Int64.(floor.((sample_extent .+ tile_size) ./ tile_size))
    # TODO visualize tile bounds to see if they overlap
    width, height = n_tiles
    total_tiles = width * height - 1
    bar = Progress(total_tiles, 1)
    @info "Utilizing $(Threads.nthreads()) threads"
    camera = i.camera
    filter_radius = film.filter.radius

    _tile = Point2f(0f0)
    _tb_min = sample_bounds.p_min .+ _tile .* tile_size
    _tb_max = min.(_tb_min .+ (tile_size - 1), sample_bounds.p_max)
    _tile_bounds = Bounds2(_tb_min, _tb_max)
    filmtiles = [FilmTile(film, _tile_bounds, filter_radius) for _ in 1:Threads.maxthreadid()]
    Threads.@threads for k in 0:total_tiles
        x, y = k % width, k ÷ width
        tile = Point2f(x, y)
        tb_min = sample_bounds.p_min .+ tile .* tile_size
        tb_max = min.(tb_min .+ (tile_size - 1), sample_bounds.p_max)
        if tb_min[1] < tb_max[1] && tb_min[2] < tb_max[2]
            tile_bounds = Bounds2(tb_min, tb_max)
            film_tile = filmtiles[Threads.threadid()]
            film_tile = update_bounds!(film, film_tile, tile_bounds)
            sample_kernel(i, camera, scene, film, film_tile, tile_bounds)
        end
        next!(bar)
    end
    to_framebuffer!(film, 1f0)
end

function get_material(bvh::BVHAccel, shape::Triangle)
    materials = bvh.materials
    @inbounds if shape.material_idx == 0
        return materials[1]
    else
        return materials[shape.material_idx]
    end
end
function get_material(scene::Scene, shape::Triangle)
    get_material(scene.aggregate, shape)
end

function only_light(lights, ray)
    l = RGBSpectrum(0.0f0)
    Base.Cartesian.@nexprs 8 i -> begin
        if i <= length(lights)
            light = lights[i]
            l += le(light, ray)
        end
    end
    return l
end

@inline function light_contribution(l, lights, wo, scene, bsdf, sampler, si)
    core = si.core
    n = si.shading.n
    # Why can't I use KernelAbstraction.@unroll here, when in Trace.jl?
    # Worked just fined when the function was defined outside
    Base.Cartesian.@nexprs 8 i -> begin
        if i <= length(lights)
            @inbounds light = lights[i]
            sampled_li, wi, pdf, tester = sample_li(light, core, get_2d(sampler))
            if !(is_black(sampled_li) || pdf ≈ 0.0f0)
                f = bsdf(wo, wi)
                if !is_black(f) && unoccluded(tester, scene)
                    l += f * sampled_li * abs(wi ⋅ n) / pdf
                end
            end
        end
    end
    return l
end

function li(
    sampler, max_depth, ray::RayDifferentials, scene::Scene, depth::Int32,
)::RGBSpectrum

    l = RGBSpectrum(0.0f0)
    # Find closest ray intersection or return background radiance.
    hit, shape, si = intersect!(scene, ray)
    lights = scene.lights
    if !hit
        return only_light(lights, ray)
    end
    # Compute emmited & reflected light at ray intersection point.
    # Initialize common variables for Whitted integrator.
    core = si.core
    n = si.shading.n
    wo = core.wo
    # Compute scattering functions for surface interaction.
    si = compute_differentials(si, ray)
    m = get_material(scene, shape)
    if m.type === NO_MATERIAL
        return li(
            sampler, max_depth, RayDifferentials(spawn_ray(si, ray.d)),
            scene, depth,
        )
    end
    bsdf = m(si, false, Radiance)
    # Compute emitted light if ray hit an area light source.
    l += le(si, wo)
    # Add contribution of each light source.
    l = light_contribution(l, lights, wo, scene, bsdf, sampler, si)
    if depth + 1 ≤ max_depth
        # Trace rays for specular reflection & refraction.
        l += specular_reflect(bsdf, sampler, max_depth, ray, si, scene, depth)
        l += specular_transmit(bsdf, sampler, max_depth, ray, si, scene, depth)
    end
    l
end

@inline function specular_reflect(
    bsdf, sampler, max_depth, ray::RayDifferentials,
    si::SurfaceInteraction, scene::Scene, depth::Int32,
)

    # Compute specular reflection direction `wi` and BSDF value.
    wo = si.core.wo
    type = BSDF_REFLECTION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        bsdf, wo, get_2d(sampler), type,
    )
    # Return contribution of specular reflection.
    ns = si.shading.n
    if !(pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0)
        return RGBSpectrum(0.0f0)
    end
    # # Compute ray differential for specular reflection.
    rd = RayDifferentials(spawn_ray(si, wi))
    if ray.has_differentials
        rx_origin = si.core.p + si.∂p∂x
        ry_origin = si.core.p + si.∂p∂y
        # Compute differential reflected directions.
        ∂n∂x = (
            si.shading.∂n∂u * si.∂u∂x
            +
            si.shading.∂n∂v * si.∂v∂x
        )
        ∂n∂y = (
            si.shading.∂n∂u * si.∂u∂y
            +
            si.shading.∂n∂v * si.∂v∂y
        )
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        rx_direction = wi - ∂wo∂x + 2.0f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
        ry_direction = wi - ∂wo∂y + 2.0f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
        rd = RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
    end
    return f * li(sampler, max_depth, rd, scene, depth + Int32(1)) * abs(wi ⋅ ns) / pdf
end

@inline function specular_transmit(
    bsdf, sampler, max_depth, ray::RayDifferentials,
    surface_intersect::SurfaceInteraction, scene::Scene, depth::Int32,
)

    # Compute specular reflection direction `wi` and BSDF value.
    wo = surface_intersect.core.wo
    type = BSDF_TRANSMISSION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        bsdf, wo, get_2d(sampler), type,
    )

    ns = surface_intersect.shading.n
    if !(pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0)
        return RGBSpectrum(0.0f0)
    end
    # TODO shift in ray direction instead of normal?
    rd = RayDifferentials(spawn_ray(surface_intersect, wi))
    if ray.has_differentials
        rx_origin = surface_intersect.core.p + surface_intersect.∂p∂x
        ry_origin = surface_intersect.core.p + surface_intersect.∂p∂y
        # Compute differential transmitted directions.
        ∂n∂x = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂x
            +
            surface_intersect.shading.∂n∂v * surface_intersect.∂v∂x
        )
        ∂n∂y = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂y
            +
            surface_intersect.shading.∂n∂v * surface_intersect.∂v∂y
        )
        # The BSDF stores the IOR of the interior of the object being
        # intersected. Compute the relative IOR by first out by assuming
        # that the ray is entering the object.
        η = 1.0f0 / bsdf.η
        if (ns ⋅ ns) < 0.0f0
            # If the ray isn't entering the object, then we need to invert
            # the relative IOR and negate the normal and its derivatives.
            η = 1.0f0 / η
            ∂n∂x, ∂n∂y, ns = -∂n∂x, -∂n∂y, -ns
        end
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        μ = η * (wo ⋅ ns) - abs(wi ⋅ ns)
        ν = η - (η * η * (wo ⋅ ns)) / abs(wi ⋅ ns)
        ∂μ∂x = ν * ∂dn∂x
        ∂μ∂y = ν * ∂dn∂y
        rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
        ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
        rd = RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
    end
    f * li(sampler, max_depth, rd, scene, depth + Int32(1)) * abs(wi ⋅ ns) / pdf
end


struct Reflect end
struct Transmit end

macro ntuple(N, value)
    expr = :(())
    for i in 1:N
        push!(expr.args, :($(esc(value))))
    end
    return expr
end

macro setindex(N, setindex_expr)
    @assert Meta.isexpr(setindex_expr, :(=))
    index_expr = setindex_expr.args[1]
    @assert Meta.isexpr(index_expr, :ref)
    tuple = index_expr.args[1]
    idx = index_expr.args[2]
    value = setindex_expr.args[2]
    expr = :(())
    for i in 1:N
        push!(expr.args, :(ifelse($i != $(esc(idx)), $(esc(tuple))[$i], $(esc(value)))))
    end
    return :($(esc(tuple)) = $expr)
end

@inline function li_iterative(
        sampler, max_depth::Int32, initial_ray::RayDifferentials, scene::S
    )::RGBSpectrum where {S<:Scene}
    accumulated_l = RGBSpectrum(0.0f0)
    # stack = MVector{8,Tuple{Trace.RayDifferentials,Int32,Trace.RGBSpectrum}}(undef)
    stack = @ntuple(8, (initial_ray, Int32(0), accumulated_l))
    pos = Int32(1)
    # stack[pos] = (initial_ray, Int32(0), accumulated_l)
    @inbounds while pos > Int32(0)
        (ray, depth, accumulated_l) = stack[pos]
        pos -= Int32(1)
        if depth == max_depth
            continue
        end
        hit, shape, si = intersect!(scene, ray)
        lights = scene.lights

        if !hit
            accumulated_l += only_light(lights, ray)
            continue
        end

        core = si.core
        wo = core.wo
        si = compute_differentials(si, ray)
        m = get_material(scene, shape)
        if m.type === NO_MATERIAL
            new_ray = RayDifferentials(spawn_ray(si, ray.d))
            pos += Int32(1)
            @setindex 8 stack[pos] = (new_ray, depth, accumulated_l)
            # stack[pos] = (new_ray, depth, accumulated_l)
            continue
        end

        bsdf = m(si, false, Radiance)
        accumulated_l += le(si, wo)
        accumulated_l = light_contribution(accumulated_l, lights, wo, scene, bsdf, sampler, si)

        if depth + 1 <= max_depth
            rd_reflect, reflect_l = specular(Reflect, bsdf, sampler, ray, si)
            if rd_reflect !== ray && pos < 8
                pos += Int32(1)
                @setindex 8 stack[pos] = (rd_reflect, depth + Int32(1), reflect_l * accumulated_l)
                # stack[pos] = (rd_reflect, depth + Int32(1), reflect_l * accumulated_l)
            end
            rd_transmit, transmit_l = specular(Transmit, bsdf, sampler, ray, si)
            if rd_transmit !== ray && pos < 8
                pos += Int32(1)
                @setindex 8 stack[pos] = (rd_transmit, depth + Int32(1), transmit_l * accumulated_l)
                # stack[pos] = (rd_transmit, depth + Int32(1), transmit_l * accumulated_l)
            end
        end
    end
    return accumulated_l
end


@inline get_type(::Type{Transmit}) = BSDF_TRANSMISSION | BSDF_SPECULAR
@inline get_type(::Type{Reflect}) = BSDF_REFLECTION | BSDF_SPECULAR

@inline function specular(
        type, bsdf, sampler, ray::RayDifferentials,
        si::SurfaceInteraction,
    )::Tuple{RayDifferentials, RGBSpectrum}

    wo = si.core.wo
    wi, f, pdf, sampled_type = sample_f(bsdf, wo, get_2d(sampler), get_type(type))

    ns = si.shading.n
    if !(pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0)
        return (ray, RGBSpectrum(0.0f0))
    end

    rd = RayDifferentials(spawn_ray(si, wi))
    if ray.has_differentials
        rd = specular_differentials(type, rd, bsdf, si, ray, wo, wi)
    end
    return rd, f * abs(wi ⋅ ns) / pdf
end

@inline function specular_differentials(::Type{Reflect}, rd, bsdf, si, ray, wo, wi)
    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    # Compute differential reflected directions.
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    ∂wo∂x = -ray.rx_direction - wo
    ∂wo∂y = -ray.ry_direction - wo
    ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
    ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
    rx_direction = wi - ∂wo∂x + 2.0f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
    ry_direction = wi - ∂wo∂y + 2.0f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
    return RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
end

@inline function specular_differentials(::Type{Transmit}, rd, bsdf, si, ray, wo, wi)

    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    # Compute differential transmitted directions.
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    # The BSDF stores the IOR of the interior of the object being
    # intersected. Compute the relative IOR by first out by assuming
    # that the ray is entering the object.
    η = 1.0f0 / bsdf.η
    if (ns ⋅ ns) < 0.0f0
        # If the ray isn't entering the object, then we need to invert
        # the relative IOR and negate the normal and its derivatives.
        η = 1.0f0 / η
        ∂n∂x, ∂n∂y, ns = -∂n∂x, -∂n∂y, -ns
    end
    ∂wo∂x = -ray.rx_direction - wo
    ∂wo∂y = -ray.ry_direction - wo
    ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
    ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
    μ = η * (wo ⋅ ns) - abs(wi ⋅ ns)
    ν = η - (η * η * (wo ⋅ ns)) / abs(wi ⋅ ns)
    ∂μ∂x = ν * ∂dn∂x
    ∂μ∂y = ν * ∂dn∂y
    rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
    ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
    return RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
end


struct Tile
    tile_indx::NTuple{2, Int32}
    width::Int32
end

# function sample_kernel_inner(i::A, scene::B, t_sampler::C, film::D, film_tile::E, camera::F, pixel::G, spp_sqr::H) where {A,B,C,D,E,F,G,H}
#     for _ in 1:t_sampler.samples_per_pixel
#         camera_sample = get_camera_sample(t_sampler, pixel)
#         ray, ω = generate_ray_differential(camera, camera_sample)
#         ray = scale_differentials(ray, spp_sqr)
#         l = RGBSpectrum(0.0f0)
#         if ω > 0.0f0
#             # l = li(t_sampler, Int32(i.max_depth), ray, scene, Int32(1))
#             l = li_iterative(t_sampler, Int32(i.max_depth), ray, scene)
#         end
#         # TODO check l for invalid values
#         if isnan(l)
#             l = RGBSpectrum(0.0f0)
#         end
#         add_sample!(film, film_tile, camera_sample.film, l, ω)
#     end
# end

@noinline function sample_tile(sampler, camera, scene, film, film_tile, tile_bounds, max_depth)
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)
    for pixel in tile_bounds
        for _ in 1:sampler.samples_per_pixel
            camera_sample = get_camera_sample(sampler, pixel)
            ray, ω = generate_ray_differential(camera, camera_sample)
            ray = scale_differentials(ray, spp_sqr)
            l = RGBSpectrum(0.0f0)
            if ω > 0.0f0
                l = li_iterative(sampler, Int32(max_depth), ray, scene)
            end
            # TODO check l for invalid values
            l = ifelse(isnan(l), RGBSpectrum(0.0f0), l)
            add_sample!(film, film_tile, camera_sample.film, l, ω)
        end
    end
    merge_film_tile!(film, film_tile)
end

function sample_tiled(scene::Scene, film)
    sample_bounds = get_sample_bounds(film)
    sample_extent = diagonal(sample_bounds)
    tile_size = 16
    n_tiles = floor.(Int64, (sample_extent .+ tile_size) ./ tile_size)
    # TODO visualize tile bounds to see if they overlap
    width, height = n_tiles
    filter_radius = film.filter.radius
    filmtiles = similar(film.pixels, tile_size * tile_size, n_tiles)
    for tile_idx in CartesianIndices((width, height))
        tile_column, tile_row = Tuple(tile_idx)
        tile_bounds = Bounds2(tb_min, tb_max)
        film_tile = update_bounds!(film, film_tile, tile_bounds)
        sample_kernel(i, camera, scene, film, film_tile, tile_bounds)
    end
    return film
end
