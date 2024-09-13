using Trace: Pixel, get_sample_bounds, diagonal, FilmTilePixel, RGBSpectrum, to_XYZ
using StructArrays
using GLMakie
using Trace, GeometryBasics
using Trace: u_int32, Bounds2, inclusive_sides
using Trace
using Trace: XYZ_to_RGB, to_XYZ
using Trace: li_iterative, get_camera_sample, generate_ray_differential, scale_differentials
import KernelAbstractions as KA
using KernelAbstractions
using AMDGPU


function sample_kernel_inner!(
        tiles, tile, tile_column::Int32, max_depth::Int32,
        scene, t_sampler, camera, pixel, spp_sqr, filter_table, filter_radius::Point2f, resolution::Point2f
    )
    px, py = pixel
    cpix = Point2f(py, resolution[1] - px)
    for _ in t_sampler.samples_per_pixel
        camera_sample = @inline get_camera_sample(t_sampler, cpix)
        ray, ω = generate_ray_differential(camera, camera_sample)
        ray = scale_differentials(ray, spp_sqr)
        l = RGBSpectrum(0f0)
        if ω > 0.0f0
            l = li_iterative(t_sampler, max_depth, ray, scene)
        end
        # TODO check l for invalid values
        if isnan(l)
            l = RGBSpectrum(0f0)
        end
        add_sample!(
            tiles, tile, tile_column, pixel, l,
            filter_table, filter_radius, ω,
        )
    end
end

struct Whitten5{TMat<:AbstractMatrix{FilmTilePixel},PMat<:AbstractMatrix{Pixel}}
    tiles::TMat
    pixel::PMat
    sample_bounds::Bounds2
    crop_bounds::Bounds2
    resolution::Point2f
    ntiles::NTuple{2, Int32}
    tile_size::Int32
    fiter_table::Matrix{Float32}
    filter_radius::Point2f
    sampler::Trace.UniformSampler
    max_depth::Int32
end

function Trace.to_gpu(ArrayType, w::Whitten5; preserve=[])
    Whitten5(
        KA.adapt(ArrayType, w.tiles),
        KA.adapt(ArrayType, w.pixel),
        w.sample_bounds,
        w.crop_bounds,
        w.resolution,
        w.ntiles,
        w.tile_size,
        w.fiter_table,
        w.filter_radius,
        w.sampler,
        w.max_depth,
    )
end

function Whitten5(film; samples_per_pixel=8, tile_size=4, max_depth=5)
    sample_bounds = get_sample_bounds(film)
    sample_extent = diagonal(sample_bounds)
    resolution = film.resolution
    n_tiles = Int64.(floor.((sample_extent .+ tile_size) ./ tile_size))
    wtiles, htiles = n_tiles .- 1
    filter_radius = film.filter.radius
    filter_table = generate_filter_table(film.filter)
    ntiles = (wtiles) * (htiles)
    tile_size_l = tile_size * tile_size
    contrib_sum = RGBSpectrum.(zeros(Vec3f, tile_size_l, ntiles))
    filter_weight_sum = zeros(Float32, tile_size_l, ntiles)
    tiles = StructArray{FilmTilePixel}(; contrib_sum, filter_weight_sum)
    sampler = Trace.UniformSampler(samples_per_pixel)
    li = LinearIndices((wtiles, htiles))
    @assert length(li) == ntiles
    Whitten5(tiles, film.pixels, sample_bounds, film.crop_bounds, resolution, Int32.((wtiles, htiles)), Int32(tile_size), filter_table, filter_radius, sampler, Int32(max_depth))
end

"""
Render scene.
"""
function (w::Whitten5)(scene::Trace.Scene, camera)
    tiles = w.tiles
    resolution = w.resolution
    filter_table = w.fiter_table
    filter_radius = w.filter_radius
    sampler = w.sampler
    max_depth = w.max_depth
    sample_bounds = w.sample_bounds
    tile_linear = LinearIndices(w.ntiles)
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)
    Threads.@threads for tile_xy in CartesianIndices(w.ntiles)
        i, j = Int32.(Tuple(tile_xy)) .- Int32(1)
        tile_column = tile_linear[tile_xy]
        tile_start = Point2f(i, j)
        tb_min = (sample_bounds.p_min .+ tile_start .* w.tile_size) .+ 1
        tb_max = min.(tb_min .+ (w.tile_size - 1), sample_bounds.p_max)
        tile_bounds = Bounds2(tb_min, tb_max)
        for pixel in tile_bounds
            sample_kernel_inner!(tiles, tile_bounds, tile_column, max_depth, scene, sampler, camera, pixel, spp_sqr, filter_table, filter_radius, resolution)
        end
        merge_film_tile!(film.pixels, film.crop_bounds, tiles, tile_bounds, Int32(tile_column))
    end
end

function launch_trace_image!(w::Whitten5, camera, scene)
    backend = KA.get_backend(w.tiles.contrib_sum)
    kernel! = sample_kernel2!(backend)
    spp_sqr = 1.0f0 / √Float32(w.sampler.samples_per_pixel)
    static_filter_table = Mat{size(w.fiter_table)...}(w.fiter_table)
    kernel!(
        w.pixel, w.tiles, w.tile_size, w.sample_bounds, w.max_depth,
        scene, camera, w.sampler, spp_sqr,
        static_filter_table, w.filter_radius,
        w.resolution, w.crop_bounds, ndrange=w.ntiles
    )
    KA.synchronize(backend)
    return w
end

include("./../docs/code/basic-scene.jl")

begin
    # Trace.clear!(film)
    w = Whitten5(film; samples_per_pixel=8, max_depth=8)
    @time launch_trace_image!(w, cam, scene)
    Trace.to_framebuffer!(film.framebuffer, w.pixel)
end

begin
    Trace.clear!(film)
    p = []
    w_gpu = Trace.to_gpu(ROCArray, Whitten5(film; samples_per_pixel=1, max_depth=8); preserve=p)
    gpu_scene = Trace.to_gpu(ROCArray, scene; preserve=p)
    GC.@preserve p begin
        @time launch_trace_image!(w_gpu, cam, gpu_scene)
        Trace.to_framebuffer!(film.framebuffer, KA.adapt(Array, w_gpu.pixel))
    end
end




#=
@code_warntype merge_film_tile!(film.pixels, film.crop_bounds, tiles, Bounds2(Point2f(1), Point2f(16)), Int32(1))

@code_warntype add_sample!(
    tiles, Bounds2(Point2f(1), Point2f(16)), Int32(1), Point2f(1), RGBSpectrum(1f0), filter_table, Point2f(1), 1.0f0
)s
@code_warntype sample_kernel_inner!(tiles, Bounds2(Point2f(1), Point2f(16)), Int32(1), Int32(5), scene, w.sampler, cam, Point2f(1), 1f0, filter_table, filter_radius, resolution)
@code_warntype sample_kernel2!(
    film.pixels, tiles, Int32(16),
    sample_bounds,
    Int32(5), scene, cam,
    w.sampler, 1.0f0, filter_table,
    filter_radius, resolution, w.crop_bounds
)
=#
