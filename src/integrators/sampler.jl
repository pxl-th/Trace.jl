abstract type Integrator end
abstract type SamplerIntegrator <: Integrator end

struct WhittedIntegrator <: SamplerIntegrator
    camera::C where C <: Camera
    sampler::S where S <: AbstractSampler
end

function (i::I where I <: SamplerIntegrator)(scene::Scene)
    # TODO ensure that p_min starts at 1
    sample_bounds = integrator.camera.film |> get_sample_bounds
    sample_extent = integrator.camera.film |> get_physical_extension
    tile_size = 16
    n_tiles = Int64.((sample_extent .+ tile_size) ./ tile_size)

    for y in 0:n_tiles[2] - 1, x in 0:n_tiles[1] - 1
        tile = Point2f0(x, y)
        tb_min = sample_bounds.p_min .+ tile .* tile_size
        tb_max = min.(tb_min .+ tile_size, sample_bounds.p_max)
        tile_bounds = Bounds2(tb_min, tb_max)

        film_tile = FilmTile(i.camera.film, tile_bounds)
        # TODO Base.iterate & Base.length for bounds to iterate over all pixels
        # return Point2
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
    # TODO write image
end
