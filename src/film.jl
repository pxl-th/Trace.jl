mutable struct Pixel
    xyz::Point3f0
    filter_weight_sum::Float32
    splat_xyz::Point3f0
end

struct Film
    resolution::Point2f0
    """
    Subset of the image to render, bounds are inclusive and start from 1.
    """
    crop_bounds::Bounds2
    diagonal::Float32
    filter::F where F <: Filter
    filename::String
    """
    pixels in (y, x) format
    """
    pixels::Matrix{Pixel}
    filter_table_width::Int32
    """
    filter_table in (y, x) format
    """
    filter_table::Matrix{Float32}
    scale::Float32
    """
    - resolution: full resolution of the image in pixels.
    - crop_bounds: subset of the image to render in [0, 1] range.
    - diagonal: length of the diagonal of the film's physical area in mm.
    - filename: filename for the output image.
    - scale: scale factor that is applied to the samples when writing image.
    """
    function Film(
        resolution::Point2f0, crop_bounds::Bounds2, filter::F,
        diagonal::Float32, scale::Float32, filename::String,
    ) where F <: Filter
        filter_table_width = 16
        filter_table = Matrix{Float32}(undef, filter_table_width, filter_table_width)
        # Compute film image bounds.
        crop_bounds = Bounds2(
            ceil.(resolution .* crop_bounds.p_min) .+ 1f0,
            ceil.(resolution .* crop_bounds.p_max),
        )
        crop_resolution = crop_bounds |> inclusive_sides .|> Int32
        # Allocate film image storage.
        pixels = Pixel[
            Pixel(Point3f0(0f0), 0f0, Point3f0(0f0))
            for y in 1:crop_resolution[end], x in 1:crop_resolution[begin]
        ]
        # Precompute filter weight table.
        r = filter.radius ./ filter_table_width
        for y in 0:filter_table_width - 1, x in 0:filter_table_width - 1
            p = Point2f0((x + 0.5f0) * r[1], (y + 0.5f0) * r[2])
            filter_table[y + 1, x + 1] = p |> filter
        end
        new(
            resolution, crop_bounds, diagonal * 0.001f0, filter, filename,
            pixels, filter_table_width, filter_table, scale,
        )
    end
end

"""
Range of integer pixels that the `Sampler`
is responsible for generating samples for.
"""
function get_sample_bounds(f::Film)
    Bounds2(
        floor.(f.crop_bounds.p_min .+ 0.5f0 .- f.filter.radius),
        ceil.(f.crop_bounds.p_max .- 0.5f0 .+ f.filter.radius),
    )
end

"""
Extent of the film in the scene.
This is needed for realistic cameras.
"""
function get_physical_extension(f::Film)
    aspect = f.resolution[2] / f.resolution[1]
    x = sqrt(f.diagonal ^ 2 / (1 + aspect ^ 2))
    y = aspect * x
    Bounds2(Point2f0(-x / 2f0, -y / 2f0), Point2f0(x / 2f0, y / 2f0))
end

mutable struct FilmTilePixel
    contrib_sum::S where S <: Spectrum
    filter_weight_sum::Float32
end
FilmTilePixel() = FilmTilePixel(RGBSpectrum(), 0f0)

struct FilmTile
    """
    Bounds should start from 1 not 0.
    """
    bounds::Bounds2
    filter_radius::Point2f0
    inv_filter_radius::Point2f0
    filter_table::Matrix{Float32}
    filter_table_width::Int32
    pixels::Matrix{FilmTilePixel}

    function FilmTile(
        bounds::Bounds2, filter_radius::Point2f0,
        filter_table::Matrix{Float32}, filter_table_width::Int32,
    )
        tile_res = (bounds |> inclusive_sides .|> Int32)
        pixels = [FilmTilePixel() for _ in 1:tile_res[2], __ in 1:tile_res[1]]
        new(
            bounds, filter_radius, 1f0 ./ filter_radius,
            filter_table, filter_table_width,
            pixels,
        )
    end
end

"""
Bounds should start from 1 not 0.
"""
function FilmTile(f::Film, sample_bounds::Bounds2)
    p0 = ceil.(sample_bounds.p_min .- 0.5f0 .- f.filter.radius)
    p1 = floor.(sample_bounds.p_max .- 0.5f0 .+ f.filter.radius) .+ 1f0
    tile_bounds = Bounds2(p0, p1) ∩ f.crop_bounds
    FilmTile(tile_bounds, f.filter.radius, f.filter_table, f.filter_table_width)
end

"""
Add sample contribution to the film tile.

- `point::Point2f0`:
    should start from 1 not 0.
    And is relative to the film, not the film tile.
"""
function add_sample!(
    t::FilmTile, point::Point2f0, spectrum::S,
    sample_weight::Float32 = 1f0,
) where S <: Spectrum
    # Compute sample's raster bounds.
    discrete_point = point .- 0.5f0
    p0 = ceil.(discrete_point .- t.filter_radius)
    p1 = floor.(discrete_point .+ t.filter_radius) .+ 1f0
    p0 = max.(p0, max.(t.bounds.p_min, Point2f0(1f0)))
    p1 = min.(p1, t.bounds.p_max)
    # @info "point $point"
    # @info "t bounds $(t.bounds.p_min) - $(t.bounds.p_max)"
    # @info "p0 $p0 | p1 $p1"
    # Precompute x & y filter offsets.
    offsets_x = Vector{Int32}(undef, Int32(p1[1] - p0[1] + 1))
    offsets_y = Vector{Int32}(undef, Int32(p1[2] - p0[2] + 1))
    for (i, x) in enumerate(p0[1]:p1[1])
        fx = abs((x - discrete_point[1]) * t.inv_filter_radius[1] * t.filter_table_width)
        offsets_x[i] = clamp(fx |> ceil, 1, t.filter_table_width)  # TODO is clipping ok?
    end
    for (i, y) in enumerate(p0[2]:p1[2])
        fy = abs((y - discrete_point[2]) * t.inv_filter_radius[2] * t.filter_table_width)
        offsets_y[i] = clamp(fy |> floor, 1, t.filter_table_width)
    end

    # for p in t.pixels
    #     @assert all(p.contrib_sum.c .≈ 0) "Should be zero, but not: $(p.contrib_sum.c)"
    # end
    # Loop over filter support & add sample to pixel array.
    for (j, y) in enumerate(p0[2]:p1[2]), (i, x) in enumerate(p0[1]:p1[1])
        w = t.filter_table[offsets_y[j], offsets_x[i]]
        pixel = get_pixel(t, Point2f0(x, y))
        # @assert all(pixel.contrib_sum.c .≈ 0) "$(pixel.contrib_sum.c) | ($x, $y)"
        @assert sample_weight <= 1
        @assert w <= 1
        pixel.contrib_sum += spectrum * sample_weight * w
        pixel.filter_weight_sum += w
    end
end

"""
Point in (x, y) format.
"""
@inline function get_pixel(t::FilmTile, p::Point2f0)
    pp = (p .- t.bounds.p_min .+ 1f0) .|> Int32
    # @info "b $(t.bounds.p_min) | $p => $pp"
    t.pixels[pp[2], pp[1]]
end

"""
Point in (x, y) format.
"""
@inline function get_pixel(f::Film, p::Point2f0)
    pp = (p .- f.crop_bounds.p_min .+ 1f0) .|> Int32
    # @info "b $(f.crop_bounds.p_min) | $p => $pp"
    f.pixels[pp[2], pp[1]]
end

function merge_film_tile!(f::Film, ft::FilmTile)
    x_range = ft.bounds.p_min[1]:ft.bounds.p_max[1]
    y_range = ft.bounds.p_min[2]:ft.bounds.p_max[2]
    # @info x_range
    # @info y_range

    for y in y_range, x in x_range
        pixel = Point2f0(x, y)
        tile_pixel = get_pixel(ft, pixel)
        merge_pixel = get_pixel(f, pixel)
        merge_pixel.xyz += tile_pixel.contrib_sum |> to_XYZ
        merge_pixel.filter_weight_sum += tile_pixel.filter_weight_sum
    end
end

function set_image(f::Film, spectrum::Matrix{S}) where S <: Spectrum
    @assert size(f.pixels) == size(spectrum)
    for (i, p) in enumerate(f.pixels)
        p.xyz = to_XYZ(spectrum[i])
        p.filter_weight_sum = 1f0
        p.splat_xyz = Point3f0(0f0)
    end
end

function save(film::Film, splat_scale::Float32 = 1f0)
    image = Array{Float32}(undef, size(film.pixels)..., 3)
    for y in 1:size(film.pixels)[1], x in 1:size(film.pixels)[2]
        pixel = film.pixels[y, x]
        image[y, x, :] .= XYZ_to_RGB(pixel.xyz)
        # Normalize pixel with weight sum.
        filter_weight_sum = pixel.filter_weight_sum
        if filter_weight_sum != 0
            inv_weight = 1f0 / filter_weight_sum
            image[y, x, :] .= max.(0f0, image[y, x, :] .* inv_weight)
        end
        # Add splat value at pixel & scale.
        splat_rgb = XYZ_to_RGB(pixel.splat_xyz)
        image[y, x, :] .+= splat_scale .* splat_rgb
        image[y, x, :] .*= film.scale
        @info image[y, x, :]
    end
    FileIO.save(film.filename, image)
end
