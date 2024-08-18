abstract type SamplerIntegrator <: Integrator end

struct WhittedIntegrator{C<: Camera, S <: AbstractSampler} <: SamplerIntegrator
    camera::C
    sampler::S
    max_depth::Int64
end

@noinline function sample_kernel_inner(pool, i, scene, t_sampler, film, film_tile, camera, pixel, spp_sqr)
    while has_next_sample(t_sampler)
        free_all(pool) # clear memory pool
        camera_sample = get_camera_sample(t_sampler, pixel)
        ray, ω = generate_ray_differential(pool, camera, camera_sample)
        scale_differentials!(ray, spp_sqr)
        l = RGBSpectrum(0f0)
        if ω > 0.0f0
            l = li(pool, i, ray, scene, 1)
        end
        # TODO check l for invalid values
        if isnan(l)
            l = RGBSpectrum(0f0)
        end
        add_sample!(film, film_tile, camera_sample.film, l, ω)
        start_next_sample!(t_sampler)
    end
end

@noinline function sample_kernel(mempools, i, camera, scene, film, film_tile, tile_bounds)

    pool = mempools[Threads.threadid()]
    t_sampler = deepcopy(i.sampler)
    spp_sqr = 1f0 / √Float32(t_sampler.samples_per_pixel)
    for pixel in tile_bounds
        start_pixel!(t_sampler, pixel)
        sample_kernel_inner(pool, i, scene, t_sampler, film, film_tile, camera, pixel, spp_sqr)
    end
    merge_film_tile!(film, film_tile)
end

"""
Render scene.
"""
function (i::SamplerIntegrator)(scene::Scene)

    sample_bounds = get_sample_bounds(get_film(i.camera))
    sample_extent = diagonal(sample_bounds)
    tile_size = 16
    n_tiles = Int64.(floor.((sample_extent .+ tile_size) ./ tile_size))
    # TODO visualize tile bounds to see if they overlap
    width, height = n_tiles
    total_tiles = width * height - 1
    bar = Progress(total_tiles, 1)
    @info "Utilizing $(Threads.nthreads()) threads"
    mempools = [MemoryPool(round(Int, 3*16384)) for _ in 1:Threads.maxthreadid()]
    film = get_film(i.camera)
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
            sample_kernel(mempools, i, camera, scene, film, film_tile, tile_bounds)
        end
        next!(bar)
    end
    save(film)
end

function li(
        pool, i::WhittedIntegrator, ray::RayDifferentials, scene::Scene, depth::Int64,
    )::RGBSpectrum

    l = RGBSpectrum(0f0)
    # Find closest ray intersection or return background radiance.
    hit, primitive, si = intersect!(pool, scene, ray)
    if !hit
        for light in scene.lights
            l += le(light, ray)
        end
        return l
    end
    # Compute emmited & reflected light at ray intersection point.
    # Initialize common variables for Whitted integrator.
    core = si.core
    n = si.shading.n
    wo = core.wo
    # Compute scattering functions for surface interaction.
    si, bsdf = compute_scattering!(pool, primitive, si, ray)
    if bsdf.bxdfs.last == 0
        return li(
            pool, spawn_ray(pool, si, ray.d),
            scene, i.sampler, depth,
        )
    end
    # Compute emitted light if ray hit an area light source.
    l += le(si, wo)
    # Add contribution of each light source.
    for light in scene.lights
        sampled_li, wi, pdf, visibility_tester = sample_li(
            pool, light, core, get_2d(i.sampler),
        )
        (is_black(sampled_li) || pdf ≈ 0f0) && continue
        f = bsdf(wo, wi)
        if !is_black(f) && unoccluded(visibility_tester, scene)
            l += f * sampled_li * abs(wi ⋅ n) / pdf
        end
    end
    if depth + 1 ≤ i.max_depth
        # Trace rays for specular reflection & refraction.
        l += specular_reflect(pool, bsdf, i, ray, si, scene, depth)
        l += specular_transmit(pool, bsdf, i, ray, si, scene, depth)
    end
    l
end

function specular_reflect(
        pool, bsdf, i::I, ray::RayDifferentials,
        surface_intersect::SurfaceInteraction, scene::Scene, depth::Int64,
    ) where I<:SamplerIntegrator

    # Compute specular reflection direction `wi` and BSDF value.

    wo = surface_intersect.core.wo
    type = BSDF_REFLECTION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        bsdf, wo, get_2d(i.sampler), type,
    )
    # Return contribution of specular reflection.
    ns = surface_intersect.shading.n
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        return RGBSpectrum(0f0)
    end
    # Compute ray differential for specular reflection.
    rd = allocate(pool, RayDifferentials, spawn_ray(pool, surface_intersect, wi))
    if ray.has_differentials
        rd.has_differentials = true
        rd.rx_origin = surface_intersect.core.p + surface_intersect.∂p∂x
        rd.ry_origin = surface_intersect.core.p + surface_intersect.∂p∂y
        # Compute differential reflected directions.
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
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        rd.rx_direction = wi - ∂wo∂x + 2f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
        rd.ry_direction = wi - ∂wo∂y + 2f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
    end
    return f * li(pool, i, rd, scene, depth + 1) * abs(wi ⋅ ns) / pdf
end

function specular_transmit(
        pool, bsdf, i::S, ray::RayDifferentials,
        surface_intersect::SurfaceInteraction, scene::Scene, depth::Int64,
    ) where S<:SamplerIntegrator

    # Compute specular reflection direction `wi` and BSDF value.
    wo = surface_intersect.core.wo
    type = BSDF_TRANSMISSION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        bsdf, wo, get_2d(i.sampler), type,
    )

    ns = surface_intersect.shading.n
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        return RGBSpectrum(0f0)
    end
    # TODO shift in ray direction instead of normal?
    rd = allocate(pool, RayDifferentials, spawn_ray(pool, surface_intersect, wi))
    if ray.has_differentials
        rd.has_differentials = true
        rd.rx_origin = surface_intersect.core.p + surface_intersect.∂p∂x
        rd.ry_origin = surface_intersect.core.p + surface_intersect.∂p∂y
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
        η = 1f0 / bsdf.η
        if (ns ⋅ ns) < 0
            # If the ray isn't entering the object, then we need to invert
            # the relative IOR and negate the normal and its derivatives.
            η = 1f0 / η
            ∂n∂x, ∂n∂y, ns = -∂n∂x, -∂n∂y, -ns
        end
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        μ = η * (wo ⋅ ns) - abs(wi ⋅ ns)
        ν = η - (η^2 * (wo ⋅ ns)) / abs(wi ⋅ ns)
        ∂μ∂x = ν * ∂dn∂x
        ∂μ∂y = ν * ∂dn∂y
        rd.rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
        rd.ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
    end
    f * li(pool, i, rd, scene, depth + 1) * abs(wi ⋅ ns) / pdf
end
