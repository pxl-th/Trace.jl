struct Pixel
    xyz::Point3f
    filter_weight_sum::Float32
    splat_xyz::Point3f
end
Pixel() = Pixel(Point3f(0.0f0), 0.0f0, Point3f(0.0f0))
struct Film{Pixels<:AbstractMatrix{Pixel}}
    resolution::Point2f
    """
    Subset of the image to render, bounds are inclusive and start from 1.
    Format: [x, y].
    """
    crop_bounds::Bounds2
    diagonal::Float32
    filter::F where F<:Filter
    filename::String
    """
    pixels in (y, x) format
    """
    pixels::Pixels
    filter_table_width::Int32
    """
    filter_table in (y, x) format
    """
    filter_table::Matrix{Float32}
    scale::Float32
    framebuffer::Matrix{RGB{Float32}}

    """
    - resolution: full resolution of the image in pixels.
    - crop_bounds: subset of the image to render in [0, 1] range.
    - diagonal: length of the diagonal of the film's physical area in mm.
    - filename: filename for the output image.
    - scale: scale factor that is applied to the samples when writing image.
    """
    function Film(
        resolution::Point2f, crop_bounds::Bounds2, filter::F,
        diagonal::Float32, scale::Float32, filename::String,
    ) where F<:Filter
        filter_table_width = 16
        filter_table = Matrix{Float32}(undef, filter_table_width, filter_table_width)
        # Compute film image bounds.
        crop_bounds = Bounds2(
            ceil.(resolution .* crop_bounds.p_min) .+ 1f0,
            ceil.(resolution .* crop_bounds.p_max),
        )
        crop_resolution = Int32.(inclusive_sides(crop_bounds))
        # Allocate film image storage.
        pixels = StructArray{Pixel}(undef, crop_resolution[end], crop_resolution[begin])
        pixels.xyz .= (Point3f(0),)
        pixels.filter_weight_sum .= 0f0
        pixels.splat_xyz .= (Point3f(0),)
        # Precompute filter weight table.
        r = filter.radius ./ filter_table_width
        for y in 0:filter_table_width-1, x in 0:filter_table_width-1
            p = Point2f((x + 0.5f0) * r[1], (y + 0.5f0) * r[2])
            filter_table[y+1, x+1] = filter(p)
        end
        framebuffer = Matrix{RGB{Float32}}(undef, size(pixels)...)
        new{typeof(pixels)}(
            resolution, crop_bounds, diagonal * 0.001f0, filter, filename,
            pixels, filter_table_width, filter_table, scale, framebuffer
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
    x = sqrt(f.diagonal^2 / (1 + aspect^2))
    y = aspect * x
    Bounds2(Point2f(-x / 2f0, -y / 2f0), Point2f(x / 2f0, y / 2f0))
end

struct FilmTilePixel{S<:Spectrum}
    contrib_sum::S
    filter_weight_sum::Float32
end
FilmTilePixel() = FilmTilePixel(RGBSpectrum(), 0f0)

struct FilmTile{Pixels<:AbstractMatrix{<:FilmTilePixel}}
    """
    Bounds should start from 1 not 0.
    """
    bounds::Bounds2
    filter_radius::Point2f
    inv_filter_radius::Point2f
    filter_table_width::Int32
    pixels::Pixels

end

function FilmTile(
        bounds::Bounds2, filter_radius::Point2f,
        filter_table_width::Int32,
    )
    # Include some padding for over rounding (since we re-use the tiles)
    tile_res = (Int32.(inclusive_sides(bounds))) .+ 2
    contrib_sum = fill(RGBSpectrum(), tile_res[2], tile_res[1])
    filter_weight_sum = fill(0.0f0, tile_res[2], tile_res[1])
    pixels = StructArray{FilmTilePixel{RGBSpectrum}}((contrib_sum, filter_weight_sum))
    FilmTile{typeof(pixels)}(
        bounds, filter_radius, 1.0f0 ./ filter_radius,
        filter_table_width,
        pixels,
    )
end

"""
Bounds should start from 1 not 0.
"""
function FilmTile(f::Film, sample_bounds::Bounds2, radius)
    p0 = ceil.(sample_bounds.p_min .- 0.5f0 .- radius)
    p1 = floor.(sample_bounds.p_max .- 0.5f0 .+ radius) .+ 1f0
    tile_bounds = Bounds2(p0, p1) ∩ f.crop_bounds
    FilmTile(tile_bounds, radius, f.filter_table_width)
end

function reset!(tile::FilmTile)
    tile.pixels.contrib_sum .= (RGBSpectrum(0f0),)
    tile.pixels.filter_weight_sum .= 0f0
end

function update_bounds!(f::Film, tile::FilmTile, sample_bounds::Bounds2)
    reset!(tile)
    radius = tile.filter_radius
    p0 = ceil.(sample_bounds.p_min .- 0.5f0 .- radius)
    p1 = floor.(sample_bounds.p_max .- 0.5f0 .+ radius) .+ 1.0f0
    bounds = Bounds2(p0, p1) ∩ f.crop_bounds
    tile_res = (Int32.(inclusive_sides(bounds)))
    @assert all(reverse(tile_res) .<= size(tile.pixels)) "$(reverse(tile_res)) != $(size(tile.pixels)) $(sample_bounds)"
    FilmTile(bounds, radius, tile.inv_filter_radius, tile.filter_table_width, tile.pixels)
end

function filter_offset!(offsets, start, stop, discrete_point, inv_filter_radius, filter_table_width)
    @inbounds for (i, x) in enumerate(Int(start):Int(stop))
        fx = abs((x - discrete_point[1]) * inv_filter_radius * filter_table_width)
        offsets[i] = clamp(ceil(fx), 1, filter_table_width)  # TODO is clipping ok?
    end
end

@inline function filter_offset(x, discrete_point, inv_filter_radius, filter_table_width)
    fx = abs((x - discrete_point[1]) * inv_filter_radius * filter_table_width)
    return clamp(ceil(Int32, fx), Int32(1), Int32(filter_table_width))  # TODO is clipping ok?
end

function filter_offsets(start, stop, discrete_point, inv_filter_radius, filter_table_width)::NTuple{8, Int32}
    range = Int32(start):Int32(stop)
    return ntuple(8) do i
        if i <= length(range)
            filter_offset(range[i], discrete_point, inv_filter_radius, filter_table_width)::Int32
        else
            Int32(0)
        end
    end
end

"""
Add sample contribution to the film tile.

- `point::Point2f`:
    should start from 1 not 0.
    And is relative to the film, not the film tile.
"""
function add_sample!(
        film::Film, t::FilmTile, point::Point2f, spectrum::S,
        sample_weight::Float32 = 1f0,
    ) where S<:Spectrum

    # Compute sample's raster bounds.
    discrete_point = point .- 0.5f0
    p0 = ceil.(Int32, discrete_point .- t.filter_radius)
    p1 = floor.(Int32, discrete_point .+ t.filter_radius) .+ Int32(1)
    p0 = Int32.(max.(p0, max.(t.bounds.p_min, Point2{Int32}(1))))
    p1 = Int32.(min.(p1, t.bounds.p_max))
    # Precompute x & y filter offsets.
    offsets_x = filter_offsets(p0[1], p1[1], discrete_point, t.inv_filter_radius[1], t.filter_table_width)
    offsets_y = filter_offsets(p0[2], p1[2], discrete_point, t.inv_filter_radius[2], t.filter_table_width)
    # Loop over filter support & add sample to pixel array.
    pixels = t.pixels
    contrib_sum = pixels.contrib_sum
    filter_weight_sum = pixels.filter_weight_sum
    filter_table = film.filter_table
    @inbounds for (j, y) in enumerate(Int(p0[2]):Int(p1[2])), (i, x) in enumerate(Int(p0[1]):Int(p1[1]))
        w = filter_table[offsets_y[j], offsets_x[i]]
        @real_assert sample_weight <= 1
        @real_assert w <= 1
        idx = get_pixel_index(t, Point2(x, y))
        contrib_sum[idx] += spectrum * sample_weight * w
        filter_weight_sum[idx] += w
    end
end

"""
Point in (x, y) format.
"""
@inline function get_pixel_index(t::FilmTile, p::Point2)
    i1, i2 = Int32.((p .- t.bounds.p_min .+ 1))
    return CartesianIndex(i2, i1)
end

"""
Point in (x, y) format.
"""
@inline function get_pixel_index(f::Film, p::Point2)
    i1, i2 = Int32.((p .- f.crop_bounds.p_min .+ 1.0))
    return CartesianIndex(i2, i1)
end

function merge_film_tile!(f::Film, ft::FilmTile)
    x_range = Int(ft.bounds.p_min[1]):Int(ft.bounds.p_max[1])
    y_range = Int(ft.bounds.p_min[2]):Int(ft.bounds.p_max[2])
    ft_contrib_sum = ft.pixels.contrib_sum
    ft_filter_weight_sum = ft.pixels.filter_weight_sum
    f_xyz = f.pixels.xyz
    f_filter_weight_sum = f.pixels.filter_weight_sum
    @inbounds for y in y_range, x in x_range
        pixel = Point2{Int32}(x, y)
        ft_idx = get_pixel_index(ft, pixel)
        f_idx = get_pixel_index(f, pixel)
        f_xyz[f_idx] += to_XYZ(ft_contrib_sum[ft_idx])
        f_filter_weight_sum[f_idx] += ft_filter_weight_sum[ft_idx]
    end
end

function set_image!(f::Film, spectrum::Matrix{S}) where {S<:Spectrum}
    @real_assert size(f.pixels) == size(spectrum)
    f.pixels.xyz .= to_XYZ.(spectrum)
    f.pixels.filter_weight_sum .= 1.0f0
    f.pixels.splat_xyz .= (Point3f(0.0f0),)
end

function to_framebuffer!(film::Film, splat_scale::Float32 = 1f0)
    image = film.framebuffer
    xyz = film.pixels.xyz
    filter_weight_sum = film.pixels.filter_weight_sum
    splat_xyz = film.pixels.splat_xyz
    @inbounds for idx in eachindex(film.pixels)
        rgb = XYZ_to_RGB(xyz[idx])
        # Normalize pixel with weight sum.
        fws = filter_weight_sum[idx]
        if fws != 0
            inv_weight = 1.0f0 / fws
            rgb = max.(0.0f0, rgb .* inv_weight)
        end
        # Add splat value at pixel & scale.
        splat_rgb = XYZ_to_RGB(splat_xyz[idx])
        rgb = rgb .+ splat_scale .* splat_rgb
        rgb = rgb .* film.scale
        rgb = map(rgb) do c
            c = ifelse(isfinite(c), c, 0.0f0)
            return clamp(c, 0.0f0, 1.0f0)
        end
        image[idx] = RGB(rgb...)
    end
end

function save(film::Film, splat_scale::Float32 = 1f0)
    to_framebuffer!(film, splat_scale)
    FileIO.save(film.filename, @view film.framebuffer[end:-1:begin, :])
end
