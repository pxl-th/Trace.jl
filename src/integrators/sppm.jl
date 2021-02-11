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
    ϕ::RGBSpectrum
    τ::RGBSpectrum
    radius::Float32
    M::Int64
    N::Float32
    vp::VisiblePoint

    function SPPMPixel(;
        Ld::RGBSpectrum = RGBSpectrum(0f0),
        ϕ::RGBSpectrum = RGBSpectrum(0f0),
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
        pixel::SPPMPixel, next::Maybe{SPPMPixelListNode} = nothing,
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
    println("Pixel bounds $pixel_bounds")
    b_sides = pixel_bounds |> inclusive_sides
    n_pixels = Int64(b_sides[1] * b_sides[2])
    pixels = [
        SPPMPixel(radius=i.initial_search_radius)
        for y in 1:b_sides[2], x in 1:b_sides[1]
    ]
    println("Total pixels $n_pixels, pixels size $(pixels |> size)")
    inv_sqrt_spp = 1f0 / sqrt(i.n_iterations) |> Float32
    light_distribution = scene |> compute_light_power_distribution
    println("Light distribution $(light_distribution.func)")

    pixel_extent = pixel_bounds |> diagonal
    tile_size = 16
    n_tiles::Point2 = Int64.(floor.((pixel_extent .+ tile_size) ./ tile_size))

    sampler = UniformSampler(1)

    width, height = n_tiles
    total_tiles = width * height - 1
    println("Total tiles $(total_tiles + 1)")
    for iteration in 1:i.n_iterations
        println("Iteration $iteration")
        # Generate visible SPPM points.
        bar = Progress(total_tiles, 1)
        Threads.@threads for k in 0:total_tiles
            x, y = k % width, k ÷ width
            tile = Point2f0(x, y)
            tile_sampler = sampler |> deepcopy

            tb_min = pixel_bounds.p_min .+ tile .* tile_size
            tb_max = min.(tb_min .+ (tile_size - 1), pixel_bounds.p_max)
            tile_bounds = Bounds2(tb_min, tb_max)
            for pixel_point in tile_bounds
                start_pixel!(tile_sampler, pixel_point)
                # set_sample_number!(tile_sampler, iteration)

                camera_sample = get_camera_sample(tile_sampler, pixel_point)
                ray, β = generate_ray_differential(i.camera, camera_sample)
                β = β |> RGBSpectrum
                @assert !isnan(β)
                is_black(β) && continue
                scale_differentials!(ray, inv_sqrt_spp)
                # Follow camera ray path until a visible point is created.
                # Get SPPMPixel for current `pixel`.
                pixel_point = pixel_point .|> Int64
                pixel = pixels[pixel_point[2], pixel_point[1]]
                specular_bounce = false
                depth = 1
                while depth ≤ i.max_depth
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
                            bsdf=bsdf, β=β |> deepcopy,
                        )
                        break
                    end
                    # If at max depth, no need to spawn ray again.
                    depth == i.max_depth && (depth += 1; continue)
                    # Spawn ray from SPPM camera path vertex.
                    wi, f, pdf, sampled_type = sample_f(
                        bsdf, wo, tile_sampler |> get_2d, BSDF_ALL,
                    )
                    (pdf ≈ 0f0 || is_black(f)) && break
                    specular_bounce = (sampled_type & BSDF_SPECULAR) != 0
                    β *= f * abs(wi ⋅ surface_interaction.shading.n) / pdf
                    @assert !isnan(β)
                    βy = to_XYZ(β)[2]
                    if βy > 0.25f0 # TODO was <
                        continue_probability = min(1f0, βy)
                        get_1d(tile_sampler) > continue_probability && break
                        β /= continue_probability
                        @assert !isnan(β) && !isinf(β)
                    end
                    ray = RayDifferentials(spawn_ray(surface_interaction, wi))
                    depth += 1
                end
            end
            bar |> next!
        end
        # Create grid of all SPPM visible points.
        grid = Maybe{SPPMPixelListNode}[nothing for _ in 1:n_pixels]
        println("SPPM grid size $(grid |> size), grid memory $(sizeof(grid))")
        grid_bounds = Bounds3()
        # Compute grid bounds for SPPM visible points.
        max_radius, min_radius = 0f0, Inf32
        for pixel in pixels
            is_black(pixel.vp.β) && continue
            grid_bounds = (
                grid_bounds ∪ expand(Bounds3(pixel.vp.p), pixel.radius)
            )
            max_radius = max(max_radius, pixel.radius)
            min_radius = min(min_radius, pixel.radius)
        end
        println("Grid bounds $grid_bounds")
        # Compute resolution of SPPM grid in each dimension.
        diag = grid_bounds |> diagonal
        max_diag = diag |> maximum
        println("Max diag $max_diag, Max radius $max_radius, Min radius $min_radius")
        base_grid_resolution = Int32(floor(max_diag / max_radius))
        println("Base grid resolution $base_grid_resolution")
        println("Diag $diag")
        @assert base_grid_resolution > 0
        grid_resolution = max.(
            1, Int64.(floor.(base_grid_resolution .* diag ./ max_diag)),
        )
        println("Grid resolution $(grid_resolution)")
        # Add visible points to SPPM grid.
        @showprogress for pixel in pixels
            is_black(pixel.vp.β) && continue
            # Add pixel's visible point to applicable grid cells.
            shift = Point3f0(pixel.radius)
            _, p_min = to_grid(
                pixel.vp.p - shift, grid_bounds, grid_resolution,
            )
            _, p_max = to_grid(
                pixel.vp.p + shift, grid_bounds, grid_resolution,
            )
            for z in p_min[3]:p_max[3], y in p_min[2]:p_max[2], x in p_min[1]:p_max[1]
                # Add visible point to grid cell (x, y, z).
                h = hash(Point3(x, y, z), n_pixels)
                # Add `node` to the start of `grid[h]` linked list.
                node = SPPMPixelListNode(pixel, grid[h])
                grid[h] = node
            end
        end
        println("Photons per iteration $(i.photons_per_iteration)")
        # Trace photons and accumulate contributions.
        @showprogress for photon_index in 0:i.photons_per_iteration - 1
            # Follow photon path for `photon_index`.
            halton_index = UInt64(
                (iteration - 1) * i.photons_per_iteration + photon_index
            )
            halton_dim = 0
            # Choose light to shoot photon from.
            light_sample = radical_inverse(halton_dim, halton_index)
            halton_dim += 1
            light_num, light_pdf, _ = sample_discrete(
                light_distribution, light_sample,
            )
            light = scene.lights[light_num]
            # Compute sample values for photon ray leaving light source.
            u_light_0 = Point2f0(
                radical_inverse(halton_dim, halton_index),
                radical_inverse(halton_dim + 1, halton_index),
            )
            u_light_1 = Point2f0(
                radical_inverse(halton_dim + 2, halton_index),
                radical_inverse(halton_dim + 3, halton_index),
            )
            u_light_time = lerp(
                i.camera.core.core.shutter_open,
                i.camera.core.core.shutter_close,
                radical_inverse(halton_dim + 4, halton_index),
            )
            halton_dim += 5
            # TODO check that rays are correctly traced
            # Generate `photon_ray` from light source and initialize β.
            le, ray, light_normal, pdf_pos, pdf_dir = sample_le(
                light, u_light_0, u_light_1, u_light_time,
            )
            (pdf_pos ≈ 0f0 || pdf_dir ≈ 0f0 || is_black(le)) && continue
            photon_ray = ray |> RayDifferentials
            β = abs(light_normal ⋅ photon_ray.d) * le / (
                light_pdf * pdf_pos * pdf_dir
            )
            is_black(β) && continue
            # Follow photon path through scene and record intersections.
            interaction::Maybe{SurfaceInteraction} = nothing

            depth = 1
            while depth ≤ i.max_depth
                hit, tmp_interaction = intersect!(scene, photon_ray)
                !hit && break
                interaction = tmp_interaction
                if depth > 1
                    # Add photon contribution to nearby visible points.
                    in_bounds, photon_grid_index = to_grid(
                        interaction.core.p, grid_bounds, grid_resolution,
                    )
                    if in_bounds
                        h = hash(photon_grid_index, n_pixels)
                        # Add photon contribution
                        # to visible points in `grid[h]`.
                        node::Maybe{SPPMPixelListNode} = grid[h]
                        while node ≢ nothing
                            if distance_squared(
                                node.pixel.vp.p, interaction.core.p,
                            ) > (node.pixel.radius ^ 2)
                                node = node.next
                                continue
                            end
                            # Update `pixel`'s ϕ & M for nearby photon.
                            wi = -photon_ray.d
                            ϕ = β * node.pixel.vp.bsdf(node.pixel.vp.wo, wi)
                            @assert !isnan(ϕ)
                            node.pixel.ϕ += ϕ
                            node.pixel.M += 1
                            node = node.next
                        end
                    end
                end
                # Sample new photon direction.
                # Compute BSDF at photon intersection point.
                compute_scattering!(interaction, photon_ray, true, Importance)
                if interaction.bsdf ≡ nothing
                    photon_ray = spawn_ray(interaction, photon_ray.d)
                    continue
                end
                # Sample BSDF spectrum and direction `wi` for reflected photon.
                bsdf_sample = Point2f0(
                    radical_inverse(halton_dim, halton_index),
                    radical_inverse(halton_dim + 1, halton_index),
                )
                halton_dim += 2
                wi, fr, pdf, sampled_type = sample_f(
                    interaction.bsdf, -photon_ray.d, bsdf_sample, BSDF_ALL,
                )
                (is_black(fr) || pdf ≈ 0f0) && break

                β_new = β * fr * abs(wi ⋅ interaction.shading.n) / pdf
                # Possibly terminate photon path with Russian roulette.
                q = max(0f0, 1f0 - to_XYZ(β_new)[2] / to_XYZ(β)[2])
                if radical_inverse(halton_dim, halton_index) < q
                    halton_dim += 1
                    break
                end
                halton_dim += 1
                β = β_new / (1f0 - q)
                photon_ray = spawn_ray(interaction, wi) |> RayDifferentials

                depth += 1
            end
        end
        # Update pixel values from this pass's photons.
        γ = 2f0 / 3f0
        for pixel in pixels
            if pixel.M > 0
                # Update pixel photon count, search radius and τ from photons.
                N_new = pixel.N + γ * pixel.M
                radius_new = pixel.radius * √(N_new / (pixel.N + pixel.M))
                pixel.τ = (
                    (pixel.τ + pixel.vp.β * pixel.ϕ) * (radius_new ^ 2)
                    / (pixel.radius ^ 2)
                )
                pixel.N = N_new
                pixel.M = 0
                pixel.radius = radius_new
                pixel.ϕ = RGBSpectrum(0f0)
            end
            pixel.vp.β = RGBSpectrum(0f0)
            pixel.vp.bsdf = nothing
        end
        # Periodically store SPPM image in film and save it.
        if iteration % i.write_frequency == 0 || iteration == i.n_iterations
            Np = iteration * i.photons_per_iteration
            # image = Matrix{RGBSpectrum}(undef, pixels |> size)
            image = fill(RGBSpectrum(0f0), pixels |> size)
            @inbounds for (i, p) in enumerate(pixels)
                image[i] = p.Ld / iteration + p.τ / (Np * π * (p.radius ^ 2))
            end
            set_image!(i.camera |> get_film, image)
            i.camera |> get_film |> save
        end
    end
end

@inline function to_grid(
    p::Point3f0, bounds::Bounds3, grid_resolution::Point3,
)::Tuple{Bool, Point3}
    pg = Int64.(floor.(grid_resolution .* offset(bounds, p)))
    in_bounds = all(1 .≤ pg .≤ grid_resolution)
    in_bounds, clamp.(pg, 1, grid_resolution)
end

@inline function hash(p::Point3, hash_size::Int64)::Int64
    (
        ((p[1] * 73856093) ⊻ (p[2] * 19349663) ⊻ (p[3] * 83492791)) % hash_size
    ) + 1
end

function uniform_sample_one_light(
    i::SurfaceInteraction, scene::Scene, sampler::S,
)::RGBSpectrum where S <: AbstractSampler
    n_lights = scene.lights |> length
    n_lights == 0 && return RGBSpectrum(0f0)

    light_num = max(1, min(Int32(ceil(get_1d(sampler) * n_lights)), n_lights))
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
    if light_pdf > 0 && !is_black(Li)
        # Evaluate BSDF for light sampling strategy.
        f = (
            interaction.bsdf(interaction.core.wo, wi, bsdf_flags)
            * abs(wi ⋅ interaction.shading.n)
        )
        if !is_black(f)
            # Compute effect of visibility for light source sample.
            # TODO handle media (use `trace` for visibility)
            !unoccluded(visibility, scene) && (Li = RGBSpectrum(0f0);)
            if !is_black(Li)
                if is_δ_light(light.flags)
                    Ld += f * Li / light_pdf
                else
                    @assert false # TODO no non delta lights right now
                    scattering_pdf = compute_pdf(
                        interaction.bsdf, interaction.core.wo, wi, bsdf_flags,
                    )
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

@inline function compute_light_power_distribution(
    scene::Scene,
)::Maybe{Distribution1D}
    length(scene.lights) == 0 && return nothing
    Distribution1D(Float32[
        (light |> power |> to_XYZ)[2] for light in scene.lights
    ])
end
