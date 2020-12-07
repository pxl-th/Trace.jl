abstract type Integrator end
abstract type SamplerIntegrator <: Integrator end

struct WhittedIntegrator <: SamplerIntegrator
    camera::C where C <: Camera
    sampler::S where S <: AbstractSampler
    max_depth::Int64
end

"""
Render scene.
"""
function (i::I where I <: SamplerIntegrator)(scene::Scene)
    # TODO ensure that p_min starts at 1
    sample_bounds = i.camera.film |> get_sample_bounds
    sample_extent = i.camera.film |> get_physical_extension
    tile_size = 16
    n_tiles = Int64.((sample_extent .+ tile_size) ./ tile_size)

    for y in 0:n_tiles[2] - 1, x in 0:n_tiles[1] - 1
        tile = Point2f0(x, y)
        tb_min = sample_bounds.p_min .+ tile .* tile_size
        tb_max = min.(tb_min .+ tile_size, sample_bounds.p_max)
        tile_bounds = Bounds2(tb_min, tb_max)

        film_tile = FilmTile(i.camera.film, tile_bounds)
        for pixel in tile_bounds
            start_pixel!(i.sampler, pixel)
            # TODO check if pixel is inside pixel bounds
            while i.sampler |> has_next_sample
                camera_sample = get_camera_sample(i.sampler, pixel)
                ray, ω = generate_ray_differential(i.camera, camera_sample)
                scale_differentials!(ray, 1f0 / √i.sampler.samples_per_pixel)

                l = RGBSpectrum(0f0)
                ω > 0f0 && (l = li(i, ray, scene))
                # TODO check l for invalid values

                add_sample!(film_tile, camera_sample.p_film, l, ω)
                i.sampler |> start_next_sample!
            end
        end
        merge_film_tile!(i.camera.film, film_tile)
    end
    i.camera.film |> save
end

function li(
    i::WhittedIntegrator, ray::RayDifferentials, scene::Scene, depth::Int64,
)::S where S <: Spectrum
    l = RGBSpectrum(0f0)
    # Find closest ray intersection or return background radiance.
    hit, surface_interaction = intersect!(scene, ray)
    if !hit
        for light in scene.lights
            l += le(light, ray)
        end
        return l
    end
    # Compute emmited & reflected light at ray intersection point.
    # Initialize common variables for Whitted integrator.
    n = surface_interaction.shading.n
    wo = surface_interaction.core.wo
    # Compute scattering functions for surface interaction.
    compute_scattering!(surface_interaction, ray)
    if surface_interaction.bsdf isa Nothing
        return li(
            spawn_ray(surface_interaction, ray.d),
            scene, i.sampler, depth,
        )
    end
    # Compute emitted light if ray hit an area light source.
    l += le(surface_interaction, wo)
    # Add contribution of each light source.
    for light in scene.lights
        li, wi, pdf, visibility_tester = sample_li(
            light, surface_interaction.core, i.sampler |> get_2d,
        )
        (is_black(li) || pdf ≈ 0f0) && continue
        f = surface_interaction.bsdf(wo, wi)
        if !is_black(f) && unoccluded(visibility_tester, scene)
            l += f * li * abs(wi ⋅ n) / pdf
        end
    end
    if depth + 1 ≤ i.max_depth
        # Trace rays for specular reflection & refraction.
        l += specular_reflect(i, ray, surface_interaction, scene, depth) # TODO implement
        l += specular_transmit(i, ray, surface_interaction, scene, depth)
    end
    l
end

function specular_reflect(
    i::SamplerIntegrator, ray::RayDifferentials,
    surface_intersect::SurfaceInteraction, scene::Scene, depth::Int64,
)
    # Compute specular reflection direction `wi` and BSDF value.
    wo = surface_intersect.core.wo
end
