struct Pixel
    xyz::Point3f
    filter_weight_sum::Float32
    splat_xyz::Point3f
end
Pixel() = Pixel(Point3f(0.0f0), 0.0f0, Point3f(0.0f0))


function filter_offset(x, discrete_point, inv_filter_radius, filter_table_width)
    fx = abs((x - discrete_point) * inv_filter_radius * filter_table_width)
    return clamp(u_int32(ceil(fx)), Int32(1), Int32(filter_table_width))  # TODO is clipping ok?
end


function generate_filter_table(filter)
    filter_table_width = 16
    filter_table = Matrix{Float32}(undef, filter_table_width, filter_table_width)
    r = filter.radius ./ filter_table_width
    for y in 0:filter_table_width-1, x in 0:filter_table_width-1
        p = Point2f((x + 0.5f0) * r[1], (y + 0.5f0) * r[2])
        filter_table[y+1, x+1] = filter(p)
    end

    point = Point2f(filter_table_width)
    # Compute sample's raster bounds.
    discrete_point = point .- 0.5f0
    # Compute sample radius around point
    p0 = ceil.(Int, discrete_point .- filter.radius)
    p1 = floor.(Int, discrete_point .+ filter.radius) .+ 1
    # Make sure we're inbounds
    inv_radius = 1.0f0 ./ filter.radius
    # Precompute x & y filter offsets.
    offsets_x = filter_offsets(p0[1], p1[1], discrete_point[1], inv_radius[1], filter_table_width)
    offsets_y = filter_offsets(p0[2], p1[2], discrete_point[2], inv_radius[2], filter_table_width)
    # Loop over filter support & add sample to pixel array.
    xrange = p0[1]:p1[1]
    yrange = p0[2]:p1[2]
    weights = zeros(Float32, length(xrange), length(yrange))
    for i in 1:length(xrange), j in 1:length(yrange)
        w = filter_table[offsets_y[j], offsets_x[i]]
        weights[i, j] = w
    end
    return weights
end

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



"""
Point in (x, y) format.
"""
@inline function get_pixel_index(crop_bounds, p::Point2)
    i1, i2 = u_int32.((p .- crop_bounds.p_min .+ 1.0f0))
    return CartesianIndex(i1, i2)
end

@inline function merge_film_tile!(f::AbstractMatrix{Pixel}, crop_bounds::Bounds2, ft::AbstractMatrix{FilmTilePixel}, tile::Bounds2, tile_col::Int32)
    ft_contrib_sum = ft.contrib_sum
    ft_filter_weight_sum = ft.filter_weight_sum
    f_xyz = f.xyz
    f_filter_weight_sum = f.filter_weight_sum
    linear = Int32(1)
    @inbounds for pixel in tile
        f_idx = get_pixel_index(crop_bounds, pixel)
        f_xyz[f_idx] += to_XYZ(ft_contrib_sum[linear, tile_col])
        f_filter_weight_sum[f_idx] += ft_filter_weight_sum[linear, tile_col]
        linear += Int32(1)
    end
    return
end

@inline function get_tile_index(bounds::Bounds2, p::Point2)
    j, i = u_int32.((p .- bounds.p_min .+ 1.0f0))
    ncols = u_int32(inclusive_sides(bounds)[1])
    return (i - Int32(1)) * ncols + j
end

@inline function add_sample!(
    tiles::AbstractMatrix{FilmTilePixel}, tile::Bounds2, tile_column::Int32, point::Point2f, spectrum::RGBSpectrum,
    filter_table, filter_radius::Point2f, sample_weight::Float32=1.0f0,
)
    # Compute sample's raster bounds.
    discrete_point = point .- 0.5f0
    # Compute sample radius around point
    p0 = u_int32.(ceil.(discrete_point .- filter_radius))
    p1 = u_int32.(floor.(discrete_point .+ filter_radius)) .+ Int32(1)
    # Make sure we're inbounds
    pmin = u_int32.(tile.p_min)
    pmax = u_int32.(tile.p_max)
    p0 = max.(p0, max.(pmin, Point2{Int32}(1)))::Point2{Int32}
    p1 = min.(p1, pmax)::Point2{Int32}
    # Loop over filter support & add sample to pixel array.
    contrib_sum = tiles.contrib_sum
    filter_weight_sum = tiles.filter_weight_sum
    xrange = p0[1]:p1[1]
    yrange = p0[2]:p1[2]
    xn = length(xrange) % Int32
    yn = length(yrange) % Int32
    @inbounds for i in Int32(1):xn, j in Int32(1):yn
        x = xrange[i]
        y = yrange[j]
        w = filter_table[i, j]
        idx = get_tile_index(tile, Point2(x, y))
        contrib_sum[idx, tile_column] += spectrum * sample_weight * w
        filter_weight_sum[idx, tile_column] += w
    end
end

function set_image!(f::Film, spectrum::Matrix{S}) where {S<:Spectrum}
    @real_assert size(f.pixels) == size(spectrum)
    f.pixels.xyz .= to_XYZ.(spectrum)
    f.pixels.filter_weight_sum .= 1.0f0
    f.pixels.splat_xyz .= (Point3f(0.0f0),)
end

function clear!(film::Film)
    film.pixels.xyz .= (Point3f(0),)
    film.pixels.filter_weight_sum .= 0.0f0
    film.pixels.splat_xyz .= (Point3f(0),)
end

function to_framebuffer!(image, pixels, scale=1f0, splat_scale::Float32=1.0f0)
    image .= RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
    xyz = pixels.xyz
    filter_weight_sum = pixels.filter_weight_sum
    splat_xyz = pixels.splat_xyz
    @inbounds for idx in eachindex(pixels)
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
        rgb = rgb .* scale
        rgb = map(rgb) do c
            return ifelse(isfinite(c), c, 0.0f0)
        end
        image[idx] = RGB(rgb...)
    end
    return image
end

function to_framebuffer!(film::Film, splat_scale::Float32 = 1f0)
    image = film.framebuffer
    to_framebuffer!(image, film.pixels, film.scale, splat_scale)
end

function save(film::Film, splat_scale::Float32 = 1f0)
    to_framebuffer!(film, splat_scale)
    FileIO.save(film.filename, @view film.framebuffer[end:-1:begin, :])
    film.framebuffer
end
