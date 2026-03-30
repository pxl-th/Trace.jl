# GPU-safe integer conversion helpers (avoid InexactError)
u_int32(x) = Base.unsafe_trunc(Int32, x)
u_int(x) = Base.unsafe_trunc(Int, x)
u_uint32(x) = Base.unsafe_trunc(UInt32, x)
u_uint64(x) = Base.unsafe_trunc(UInt64, x)
floor_int32(x) = Base.unsafe_trunc(Int32, floor(x))
floor_int(x) = Base.unsafe_trunc(Int, floor(x))
round_int32(x) = Base.unsafe_trunc(Int32, round(x))

struct Pixel
    xyz::Point3f
    filter_weight_sum::Float32
    splat_xyz::Point3f
end
Pixel() = Pixel(Point3f(0.0f0), 0.0f0, Point3f(0.0f0))


struct FilmTilePixel{S<:Spectrum}
    contrib_sum::S
    filter_weight_sum::Float32
end
FilmTilePixel() = FilmTilePixel(RGBSpectrum(), 0.0f0)

struct Film{
    Pixels<:AbstractMatrix{Pixel},
    Tiles<:AbstractMatrix{FilmTilePixel},
    FB<:AbstractMatrix{RGB{Float32}},
    PP<:AbstractMatrix{RGBA{Float32}},
    AB<:AbstractMatrix{RGB{Float32}},
    NB<:AbstractMatrix{Vec3f},
    DB<:AbstractMatrix{Float32},
}
    resolution::Point2f
    """
    Subset of the image to render, bounds are inclusive and start from 1.
    Format: [x, y].
    """
    crop_bounds::Bounds2
    diagonal::Float32
    """
    pixels in (y, x) format
    """
    pixels::Pixels

    tiles::Tiles
    tile_size::Int32
    ntiles::NTuple{2, Int32}
    """
    filter_table in (y, x) format
    """
    filter_table::Matrix{Float32}
    filter_table_width::Int32
    filter_radius::Point2f
    filter_params::GPUFilterParams
    scale::Float32

    # Raw rendered output (HDR)
    framebuffer::FB

    # Auxiliary buffers for denoising (first-hit data)
    albedo::AB
    normal::NB
    depth::DB

    # Postprocessed output (overwritten each postprocess! call) — RGBA for direct GPU blit
    postprocess::PP

    # Render state - tracks iteration/sample progress for progressive rendering
    iteration_index::Base.RefValue{Int32}
end


"""
- resolution: full resolution of the image in pixels.
- crop_bounds: subset of the image to render in [0, 1] range.
- diagonal: length of the diagonal of the film's physical area in mm.
- scale: scale factor that is applied to the samples when writing image.
"""
function Film(
        resolution::Point2f;
        filter=Hikari.LanczosSincFilter(Point2f(1f0), 3f0),
        crop_bounds::Bounds2=Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
        diagonal=1f0, scale=1f0,
        tile_size=4, filter_table_width=16,
    )

    # Compute film image bounds.
    crop_bounds = Bounds2(
        ceil.(resolution .* crop_bounds.p_min) .+ 1.0f0,
        ceil.(resolution .* crop_bounds.p_max),
    )
    crop_resolution = Int32.(inclusive_sides(crop_bounds))
    # Allocate film image storage.
    pixels = StructArray{Pixel}(undef, crop_resolution[end], crop_resolution[begin])
    pixels.xyz .= (Point3f(0),)
    pixels.filter_weight_sum .= 0.0f0
    pixels.splat_xyz .= (Point3f(0),)

    # Compute sample bounds for tile layout.
    sample_bounds = Bounds2(
        floor.(crop_bounds.p_min .+ 0.5f0 .- filter.radius),
        ceil.(crop_bounds.p_max .- 0.5f0 .+ filter.radius),
    )
    sample_extent = Hikari.diagonal(sample_bounds)
    resolution = resolution
    n_tiles = Int64.(floor.((sample_extent .+ tile_size) ./ tile_size))
    wtiles, htiles = n_tiles .- 1

    # Precompute filter weight table.
    filter_table = Matrix{Float32}(undef, filter_table_width, filter_table_width)
    r = filter.radius ./ filter_table_width
    for y in 0:filter_table_width-1, x in 0:filter_table_width-1
        p = Point2f((x + 0.5f0) * r[1], (y + 0.5f0) * r[2])
        filter_table[y+1, x+1] = filter(p)
    end
    ntiles = wtiles * htiles
    tile_size_l = tile_size * tile_size
    contrib_sum = RGBSpectrum.(zeros(Vec3f, tile_size_l, ntiles))
    filter_weight_sum = zeros(Float32, tile_size_l, ntiles)
    tiles = StructArray{FilmTilePixel}(; contrib_sum, filter_weight_sum)

    pixel_size = size(pixels)
    framebuffer = Matrix{RGB{Float32}}(undef, pixel_size...)

    # Auxiliary buffers for denoising
    albedo = fill(RGB{Float32}(0, 0, 0), pixel_size...)
    normal = fill(Vec3f(0, 0, 0), pixel_size...)
    depth = fill(0.0f0, pixel_size...)

    # Postprocess target buffer
    postprocess = Matrix{RGBA{Float32}}(undef, pixel_size...)

    return Film(
        resolution,
        crop_bounds,
        diagonal * 0.001f0,
        pixels,
        tiles, Int32(tile_size), (Int32(wtiles), Int32(htiles)),
        filter_table,
        Int32(filter_table_width),
        filter.radius,
        GPUFilterParams(filter),
        scale,
        framebuffer,
        albedo,
        normal,
        depth,
        postprocess,
        Ref(Int32(0)),  # iteration_index starts at 0
    )
end


function clear!(film::Film)
    # Reset iteration counter for progressive rendering
    film.iteration_index[] = Int32(0)

    # Clear film buffers
    film.tiles.contrib_sum .= (RGBSpectrum(0.0f0),)
    film.tiles.filter_weight_sum .= 0.0f0
    film.pixels.xyz .= (Point3f(0),)
    film.pixels.filter_weight_sum .= 0.0f0
    film.pixels.splat_xyz .= (Point3f(0),)

    # Clear auxiliary buffers
    film.albedo .= (RGB{Float32}(0, 0, 0),)
    film.normal .= (Vec3f(0, 0, 0),)
    film.depth .= 0.0f0
end

@kernel inbounds=true function film_to_rgb!(image, xyz, filter_weight_sum, splat_xyz, scale, splat_scale)
    idx = @index(Global)
     begin
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
            ifelse(isfinite(c), c, 0.0f0)
        end
        image[idx] = RGB(rgb...)
    end
    nothing
end

function to_framebuffer!(image, pixels, scale=1f0, splat_scale::Float32=1.0f0)
    image .= RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
    xyz = pixels.xyz
    filter_weight_sum = pixels.filter_weight_sum
    splat_xyz = pixels.splat_xyz
    backend = KA.get_backend(image)
    kernel! = film_to_rgb!(backend)
    kernel!(image, xyz, filter_weight_sum, splat_xyz, scale, splat_scale, ndrange=length(image))
    KA.synchronize(backend)
    return image
end

function to_framebuffer!(film::Film, splat_scale::Float32 = 1f0)
    image = film.framebuffer
    to_framebuffer!(image, film.pixels, film.scale, splat_scale)
end

# ============================================================================
# Auxiliary Buffer Filling
# ============================================================================
# Separate kernel to fill albedo/normal/depth from primary ray hits.
# This is independent of the main integrator and can be called optionally.

"""
    fill_aux_buffers!(film, scene, camera)

Fill auxiliary buffers (albedo, normal, depth) by tracing primary rays.
Uses KernelAbstractions for GPU compatibility.

This traces one ray per pixel (center of pixel) and stores first-hit data.
Should be called before or after main rendering - the auxiliary buffers
are used for denoising in postprocess!.
"""
function fill_aux_buffers!(film::Film, scene, camera; has_infinite_lights::Bool=false)
    albedo = film.albedo
    normal = film.normal
    depth = film.depth
    resolution = film.resolution
    crop_bounds = film.crop_bounds

    # Depth for rays that miss geometry: Inf32 means "escaped" (masked in postprocess),
    # a large finite value means "hit infinite light" (not masked).
    miss_depth = has_infinite_lights ? Float32(1e30) : Inf32

    backend = KA.get_backend(albedo)
    kernel! = aux_buffer_kernel!(backend)
    # Pass scene.accel directly instead of the full scene struct to avoid
    # misaligned address errors on CUDA from passing large nested structs.
    kernel!(
        albedo, normal, depth,
        resolution, crop_bounds,
        scene.accel, camera, miss_depth;
        ndrange = length(albedo)
    )
    KA.synchronize(backend)
    return film
end

@kernel inbounds=true function aux_buffer_kernel!(albedo, normal, depth, resolution, crop_bounds, accel, camera, miss_depth::Float32)
    idx = @index(Global)

    # Convert linear index to 2D pixel coordinates
    # albedo is (height, width) = (rows, cols)
    h, _ = size(albedo)
    row = ((idx - 1) % h) + 1
    col = ((idx - 1) ÷ h) + 1

    # Pixel coords (x=col, y=row) - match sampler integrator convention
    px = Float32(col) + crop_bounds.p_min[1] - 1f0
    py = Float32(row) + crop_bounds.p_min[2] - 1f0
    pixel = Point2f(px + 0.5f0, py + 0.5f0)  # Center of pixel

    # Generate primary ray (use generate_ray, not generate_ray_differential —
    # aux buffers don't need ray differentials, and the simpler call avoids
    # large return-value stack pressure on CUDA)
    camera_sample = CameraSample(pixel, Point2f(0.5f0), 0f0)
    ray, ω = generate_ray(camera, camera_sample)

     if ω > 0f0
        # Trace primary ray
        hit, _primitive, si = intersect!(accel, ray)

        if hit
            # Store normal (world space)
            normal[idx] = Vec3f(si.core.n)

            # Store depth (distance from camera)
            cam_pos = ray.o
            hit_pos = si.core.p
            depth[idx] = sqrt(sum((hit_pos .- cam_pos) .^ 2))

            # Get albedo from material
            # Use white as default, material-specific albedo requires material dispatch
            albedo[idx] = RGB{Float32}(0.8f0, 0.8f0, 0.8f0)
        else
            # No geometry hit — miss_depth is Inf32 (escaped) or 1e30 (infinite light)
            normal[idx] = Vec3f(0f0, 0f0, 0f0)
            depth[idx] = miss_depth
            albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
        end
    else
        # Invalid ray
        normal[idx] = Vec3f(0f0, 0f0, 0f0)
        depth[idx] = Inf32
        albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
    end
end

function save(film::Film, splat_scale::Float32 = 1f0)
    to_framebuffer!(film, splat_scale)
    FileIO.save(film.filename, @view film.framebuffer[end:-1:begin, :])
    film.framebuffer
end

# ============================================================================
# Resource Cleanup
# ============================================================================

"""
    free!(film::Film)

Release GPU memory held by the film by triggering finalizers on all arrays.
"""
function free!(film::Film)
    finalize(film.pixels)
    finalize(film.tiles)
    finalize(film.filter_table)
    finalize(film.framebuffer)
    finalize(film.albedo)
    finalize(film.normal)
    finalize(film.depth)
    finalize(film.postprocess)
    return nothing
end
