mutable struct VisiblePoint
    p::Point3f0
    wo::Vec3f0
    bsdf::Maybe{BSDF}
    β::RGBSpectrum

    function VisiblePoint(;
        p::Point3f0 = Point3f0(0f0), wo::Vec3f0 = Vec3f0(0f0),
        bsdf::Maybe{BSDF} = nothing, β::RGBSpectrum = RGBSpectrum(0f0),
    )
        new(p, wo, bsdf, β)
    end
end

mutable struct SPPMPixel
    Ld::RGBSpectrum
    ϕ::Vec3f0
    τ::RGBSpectrum
    radius::Float32
    M::Int64
    N::Float32
    vp::VisiblePoint

    function SPPMPixel(;
        Ld::RGBSpectrum = RGBSpectrum(0f0), ϕ::Vec3f0 = Vec3f0(0f0),
        τ::RGBSpectrum = RGBSpectrum(0f0),
        radius::Float32 = 0f0, M::Int64 = 0, N::Int64 = 0,
        vp::VisiblePoint = VisiblePoint()
    )
        new(Ld, ϕ, τ, radius, M, N, vp)
    end
end

mutable struct SPPMPixelListNode
    pixel::SPPMPixel
    next::Maybe{SPPMPixelListNode}

    function SPPMPixelListNode(
        pixel::SPPMPixel = SPPMPixel(),
        next::Maybe{SPPMPixelListNode} = nothing,
    )
        new(pixel, next)
    end
end

struct SPPMIntegrator <: Integrator
    camera::C where C <: Camera
    initial_search_radius::Float32
    max_depth::Int64
    n_iterations::Int64
    photons_per_iteration::Int64
    write_frequency::Int64

    function SPPMIntegrator(
        camera::C, initial_search_radius::Float32, max_depth::Int64,
        n_iterations::Int64, photons_per_iteration::Int64 = -1,
        write_frequency::Int64 = 100,
    ) where C <: Camera
        photons_per_iteration = (
            photons_per_iteration > 0
            ? photons_per_iteration : area(get_film(camera).crop_bounds)
        )
        new(
            camera, initial_search_radius, max_depth,
            n_iterations, photons_per_iteration, write_frequency,
        )
    end
end

function (i::SPPMIntegrator)(scene::Scene)
    pixel_bounds = get_film(i.camera).crop_bounds
    b_sides = pixel_bounds |> inclusive_sides # TODO maybe regular sides?
    n_pixels = b_sides[1] * b_sides[2]
    pixels = [
        SPPMPixel(radius=i.initial_search_radius)
        for y in 1:b_sides[2], x in 1:b_sides[1]
    ]
    inv_sqrt_spp = 1f0 / sqrt(i.n_iterations) |> Float32

    pixel_extent = pixel_bounds |> diagonal
    tile_size = 16
    n_tiles::Point2 = Int64.(floor.((pixel_extent .+ tile_size) ./ tile_size))

    sampler = UniformSampler(1)

    width, height = n_tiles
    total_tiles = width * height - 1
    for iteration in 1:i.n_iterations
        # Generate visible SPPM points.
        for k in 0:total_tiles
            x, y = k % width, k ÷ width
            tile = Point2f0(x, y)
            tile_sampler = sampler |> deepcopy

            tb_min = pixel_bounds.p_min .+ tile .* tile_size
            tb_max = min.(tb_min .+ tile_size, pixel_bounds.p_max)
            tile_bounds = Bounds2(tb_min, tb_max)
            for pixel_point in tile_bounds
                start_pixel!(tile_sampler, pixel_point)
                # set_sample_number!(tile_sampler, iteration)

                camera_sample = get_camera_sample(tile_sampler, pixel_point)
                ray, β = generate_ray_differential(i.camera, camera_sample)
                β = β |> RGBSpectrum
                is_black(β) && continue
                scale_differentials!(ray, inv_sqrt_spp)
                # Follow camera ray path until a visible point is created.
                # Get SPPMPixel for current `pixel`.
                p = Int64.(pixel_point .- pixel_bounds.p_min .+ 1)
                pixel = pixels[p[2], p[1]]
                specular_bounce = false
                depth = 1
                while depth < i.max_depth
                    hit, surface_interaction = intersect!(scene, ray)
                    if !hit # Accumulate light contributions to the background.
                        for light in scene.lights
                            pixel.Ld += β * le(light, ray)
                        end
                        break
                    end
                    # Process SPPM camera ray intersection.
                    # Compute BSDF at SPPM camera ray intersection.
                    compute_scattering!(surface_interaction, ray, true)
                    if surface_interaction.bsdf ≡ nothing
                        ray = spawn_ray(surface_interaction, ray.d)
                        depth -= 1
                        continue
                    end
                    bsdf = surface_interaction.bsdf
                    # Accumulate direct illumination
                    # at SPPM camera ray intersection.
                    wo = -ray.d
                    if depth == 1 || specular_bounce
                        pixel.Ld += β * le(surface_interaction, wo)
                    end
                    pixel.Ld += uniform_sample_one_light(
                        surface_interaction, scene, tile_sampler,
                    )
                    # Possibly create visible point and end camera path.
                    is_diffuse = num_components(bsdf,
                        BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE,
                    ) > 0
                    is_glossy = num_components(bsdf,
                        BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION
                    ) > 0
                    if is_diffuse || (is_glossy && depth == i.max_depth)
                        pixel.vp = VisiblePoint(
                            p=surface_interaction.core.p, wo=wo,
                            bsdf=bsdf, β=β,
                        )
                        break
                    end
                    depth == i.max_depth && continue
                    # Spawn ray from SPPM camera path vertex.
                    wi, f, pdf, sampled_type = sample_f(
                        bsdf, wo, tile_sampler |> get_2d, BSDF_ALL,
                    )
                    (pdf ≈ 0f0 || is_black(f)) && break
                    specular_bounce = (sampled_type & BSDF_SPECULAR) != 0
                    β *= f * abs(wi ⋅ surface_interaction.shading.n) / pdf
                    βy = to_XYZ(β)[2]
                    if βy < 0.25f0
                        continue_probability = min(1f0, βy)
                        get_1d(tile_sampler) > continue_probability && break
                        β /= continue_probability
                    end
                    ray = RayDifferentials(spawn_ray(surface_interaction, wi))
                end
            end
        end
        # Create grid of all SPPM visible points.
        grid = SPPMPixelListNode[SPPMPixelListNode() for _ in 1:n_pixels]
        grid_bounds = Bounds3()
        # Compute grid bounds for SPPM visible points.
        max_radius = 0f0
        for pixel in pixels
            is_black(pixel.vp.β) && continue
            grid_bounds = (
                grid_bounds ∪ expand(Bounds3(pixel.vp.p), pixel.radius)
            )
            max_radius = max(max_radius, pixel.radius)
        end
        # Compute resolution of SPPM grid in each dimension.
        diag = grid_bounds |> diagonal
        max_diag = diag |> maximum
        base_grid_resolution = Int32(floor(max_diag / max_radius))
        @assert base_grid_resolution > 0
        grid_resolution = max.(
            1, Int64.(floor.(base_grid_resolution .* diag ./ max_radius)),
        )
        # Add visible points to SPPM grid.
        # Trace photons and accumulate contributions.
        # Update pixel values from this pass's photons.
        # Periodically store SPPM image in film and save it.
    end
end

function uniform_sample_one_light(
    i::SurfaceInteraction, scene::Scene, sampler::S,
)::RGBSpectrum where S <: AbstractSampler
    n_lights = scene.lights |> length
    n_lights == 0 && return RGBSpectrum(0f0)

    light_num = max(1, min(Int32(ceil(get_1d(sampler)) * n_lights), n_lights))
    light_pdf = 1f0 / n_lights

    light = scene.lights[light_num]
    u_light = sampler |> get_2d
    u_scatter = sampler |> get_2d

    estimate_direct(i, u_scatter, light, u_light, scene, sampler) / light_pdf
end

function estimate_direct(
    interaction::SurfaceInteraction, u_scatter::Point2f0, light::L,
    u_light::Point2f0, scene::Scene, sampler::S, specular::Bool = false,
)::RGBSpectrum where {L <: Light, S <: AbstractSampler}
    bsdf_flags = specular ? BSDF_ALL : (BSDF_ALL & ~BSDF_SPECULAR)
    Ld = RGBSpectrum(0f0)
    # Sample light source with multiple importance sampling.
    Li, wi, light_pdf, visibility = sample_li(light, interaction.core, u_light)
    scattering_pdf = 0
    if light_pdf > 0 && !is_black(Li)
        # Evaluate BSDF for light sampling strategy.
        f = (
            interaction.bsdf(interaction.core.wo, wi, bsdf_flags)
            * abs(wi ⋅ interaction.shading.n)
        )
        scattering_pdf = compute_pdf(
            interaction.bsdf, interaction.core.wo, wi, bsdf_flags,
        )
        if !is_black(f)
            # Compute effect of visibility for light source sample.
            !unoccluded(visibility, scene) && (Li = RGBSpectrum(0f0);)
            if !is_black(Li)
                if is_δ_light(light.flags)
                    Ld += f * Li / light_pdf
                else
                    weight = power_heuristic(1, light_pdf, 1, scattering_pdf)
                    Ld += f * Li * weight / light_pdf
                end
            end
        end
    end
    # TODO Sample BSDF with multiple importance sampling.
    # This requires non-δ light sources.
    Ld
end

@inline function power_heuristic(
    nf::Int64, f_pdf::Float32, ng::Int64, g_pdf::Float32,
)
    f = (nf * f_pdf) ^ 2
    g = (ng * g_pdf) ^ 2
    f / (f + g)
end
