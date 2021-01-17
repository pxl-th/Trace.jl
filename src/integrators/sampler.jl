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
    sample_bounds = i.camera |> get_film |> get_sample_bounds
    sample_extent = sample_bounds |> diagonal
    tile_size = 16
    n_tiles::Point2 = Int64.(floor.((sample_extent .+ tile_size) ./ tile_size))

    # TODO visualize tile bounds to see if they overlap
    for y in 0:n_tiles[2] - 1, x in 0:n_tiles[1] - 1
        tile = Point2f0(x, y)
        tb_min = sample_bounds.p_min .+ tile .* tile_size
        tb_max = min.(tb_min .+ tile_size, sample_bounds.p_max)
        tile_bounds = Bounds2(tb_min, tb_max)

        film_tile = FilmTile(i.camera |> get_film, tile_bounds)
        for pixel in tile_bounds
            start_pixel!(i.sampler, pixel)
            # TODO check if pixel is inside pixel bounds
            while i.sampler |> has_next_sample
                camera_sample = get_camera_sample(i.sampler, pixel)
                # camera_sample = CameraSample(Point2f0(0), Point2f0(0), 0f0)
                ray, ω = generate_ray_differential(i.camera, camera_sample)
                scale_differentials!(
                    ray, 1f0 / √Float32(i.sampler.samples_per_pixel),
                )
                # println("Casting ray: $(ray.o) -> $(ray.d)")

                l = RGBSpectrum(0f0)
                ω > 0f0 && (l = li(i, ray, scene, 1);)
                # TODO check l for invalid values
                isnan(l) && (l = RGBSpectrum(0f0);)
                # exit()

                add_sample!(film_tile, camera_sample.film, l, ω)
                i.sampler |> start_next_sample!
            end
        end
        merge_film_tile!(i.camera |> get_film, film_tile)
    end
    i.camera |> get_film |> save
end

function li(
    i::WhittedIntegrator, ray::RayDifferentials, scene::Scene, depth::Int64,
)::RGBSpectrum
    # println("Integrator LI [depth=$depth]")
    # println("\t- ray $(ray.o) -> $(ray.d)")
    l = RGBSpectrum(0f0)
    # Find closest ray intersection or return background radiance.
    hit, surface_interaction = intersect!(scene, ray)
    if !hit
        for light in scene.lights
            l += le(light, ray)
        end
        # println("\t[x] no hit")
        return l
    end
    # println("\t- hit at $(surface_interaction.core.p) from $(surface_interaction.core.wo) direction")
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
        sampled_li, wi, pdf, visibility_tester = sample_li(
            light, surface_interaction.core, i.sampler |> get_2d,
        )
        (is_black(sampled_li) || pdf ≈ 0f0) && continue
        f = surface_interaction.bsdf(wo, wi)
        # println("\t- BSDF f $f, unoccluded $(unoccluded(visibility_tester, scene))")
        if !is_black(f) && unoccluded(visibility_tester, scene)
            l += f * sampled_li * abs(wi ⋅ n) / pdf
        end
    end
    if depth + 1 ≤ i.max_depth
        # Trace rays for specular reflection & refraction.
        l += specular_reflect(i, ray, surface_interaction, scene, depth)
        l += specular_transmit(i, ray, surface_interaction, scene, depth)
    end
    # println("[♥ | depth=$depth] return l=$l")
    l
end

function specular_reflect(
    i::I, ray::RayDifferentials,
    surface_intersect::SurfaceInteraction, scene::Scene, depth::Int64,
) where I <: SamplerIntegrator
    # Compute specular reflection direction `wi` and BSDF value.
    wo = surface_intersect.core.wo
    type = BSDF_REFLECTION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        surface_intersect.bsdf, wo, i.sampler |> get_2d, type,
    )
    # Return contribution of specular reflection.
    ns = surface_intersect.shading.n
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        return RGBSpectrum(0f0)
    end
    # Compute ray differential for specular reflection.
    rd = spawn_ray(surface_intersect, wi) |> RayDifferentials
    if ray.has_differentials
        rd.has_differentials = true
        rd.rx_origin = surface_intersect.core.p + surface_intersect.∂p∂x
        rd.ry_origin = surface_intersect.core.p + surface_intersect.∂p∂y
        # Compute differential reflected directions.
        ∂n∂x = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂x
            + surface_intersect.shading.∂n∂v * surface_intersect.∂v∂x
        )
        ∂n∂y = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂y
            + surface_intersect.shading.∂n∂v * surface_intersect.∂v∂y
        )
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        rd.rx_direction = wi - ∂wo∂x + 2f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
        rd.ry_direction = wi - ∂wo∂y + 2f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
    end
    f * li(i, rd, scene, depth + 1) * abs(wi ⋅ ns) / pdf
end

function specular_transmit(
    i::S, ray::RayDifferentials,
    surface_intersect::SurfaceInteraction, scene::Scene, depth::Int64,
) where S <: SamplerIntegrator
    # Compute specular reflection direction `wi` and BSDF value.
    wo = surface_intersect.core.wo
    type = BSDF_TRANSMISSION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        surface_intersect.bsdf, wo, i.sampler |> get_2d, type,
    )
    # println("Specular transmit [depth=$depth]:")
    # println("\t- wo $wo")
    # println("\t- wi $wi")
    # println("\t- pdf $pdf")
    # println("\t- f $f")

    ns = surface_intersect.shading.n
    # println("\t- abs(wi ⋅ ns) $(abs(wi ⋅ ns))")
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        # println("\t[x] Specular transmit return black")
        return RGBSpectrum(0f0)
    end
    # TODO shift in ray direction instead of normal?
    rd = spawn_ray(surface_intersect, wi ) |> RayDifferentials
    if ray.has_differentials
        rd.has_differentials = true
        rd.rx_origin = surface_intersect.core.p + surface_intersect.∂p∂x
        rd.ry_origin = surface_intersect.core.p + surface_intersect.∂p∂y
        # Compute differential transmitted directions.
        ∂n∂x = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂x
            + surface_intersect.shading.∂n∂v * surface_intersect.∂v∂x
        )
        ∂n∂y = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂y
            + surface_intersect.shading.∂n∂v * surface_intersect.∂v∂y
        )
        # The BSDF stores the IOR of the interior of the object being
        # intersected. Compute the relative IOR by first out by assuming
        # that the ray is entering the object.
        η = 1f0 / surface_intersect.bsdf.η
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
        ν = η - (η ^ 2 * (wo ⋅ ns)) / abs(wi ⋅ ns)
        ∂μ∂x = ν * ∂dn∂x
        ∂μ∂y = ν * ∂dn∂y
        rd.rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
        rd.ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
    end
    f * li(i, rd, scene, depth + 1) * abs(wi ⋅ ns) / pdf
end
