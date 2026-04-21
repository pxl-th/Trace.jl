# GPU-safe integer conversion helpers (avoid InexactError)
u_int32(x) = Base.unsafe_trunc(Int32, x)
u_int(x) = Base.unsafe_trunc(Int, x)
u_uint32(x) = Base.unsafe_trunc(UInt32, x)
u_uint64(x) = Base.unsafe_trunc(UInt64, x)
floor_int32(x) = Base.unsafe_trunc(Int32, floor(x))
floor_int(x) = Base.unsafe_trunc(Int, floor(x))
round_int32(x) = Base.unsafe_trunc(Int32, round(x))

struct Film{
    FB<:AbstractMatrix{RGB{Float32}},
    PP<:AbstractMatrix{RGBA{Float32}},
    AB<:AbstractMatrix{RGB{Float32}},
    NB<:AbstractMatrix{Vec3f},
    DB<:AbstractMatrix{Float32},
}
    resolution::Point2f
    crop_bounds::Bounds2
    diagonal::Float32

    # Filter parameters (for pixel reconstruction in wavefront integrator)
    filter_table::Matrix{Float32}
    filter_table_width::Int32
    filter_radius::Point2f
    filter_params::GPUFilterParams
    scale::Float32

    # Sensor-calibrated linear HDR output (written by integrator)
    # Contains raw linear sRGB with imaging_ratio already applied.
    # Equivalent to pbrt-v4's GetPixelRGB output.
    framebuffer::FB

    # Auxiliary buffers for denoising (first-hit data)
    albedo::AB
    normal::NB
    depth::DB

    # Display-ready output (written by postprocess!)
    postprocess::PP

    # Progressive rendering state
    iteration_index::Base.RefValue{Int32}

    # Persistent scratch buffers for HW RT fill_aux_buffers!.
    # Cached on the film so they survive across frames; freed in free!(film).
    # `Any` to avoid leaking RTRay/RTHitResult types into Film's type parameters.
    aux_rays::Base.RefValue{Any}
    aux_results::Base.RefValue{Any}
end

"""
    Film(resolution; filter, crop_bounds, diagonal, scale, filter_table_width)

Create a film buffer for rendering.

- `resolution`: Image size in pixels, e.g. `Point2f(1920, 1080)`
- `filter`: Pixel reconstruction filter (default: LanczosSinc)
- `crop_bounds`: Region to render in [0,1] range (default: full image)
"""
function Film(
        resolution::Point2f;
        filter=Hikari.LanczosSincFilter(Point2f(1f0), 3f0),
        crop_bounds::Bounds2=Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
        diagonal=1f0, scale=1f0,
        tile_size=4, filter_table_width=16,
    )
    crop_bounds = Bounds2(
        ceil.(resolution .* crop_bounds.p_min) .+ 1.0f0,
        ceil.(resolution .* crop_bounds.p_max),
    )
    crop_resolution = Int32.(inclusive_sides(crop_bounds))

    # Precompute filter weight table
    filter_table = Matrix{Float32}(undef, filter_table_width, filter_table_width)
    r = filter.radius ./ filter_table_width
    for y in 0:filter_table_width-1, x in 0:filter_table_width-1
        p = Point2f((x + 0.5f0) * r[1], (y + 0.5f0) * r[2])
        filter_table[y+1, x+1] = filter(p)
    end

    pixel_size = (crop_resolution[end], crop_resolution[begin])
    framebuffer = Matrix{RGB{Float32}}(undef, pixel_size...)
    albedo = fill(RGB{Float32}(0, 0, 0), pixel_size...)
    normal = fill(Vec3f(0, 0, 0), pixel_size...)
    depth = fill(0.0f0, pixel_size...)
    postprocess = Matrix{RGBA{Float32}}(undef, pixel_size...)

    return Film(
        resolution,
        crop_bounds,
        diagonal * 0.001f0,
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
        Ref(Int32(0)),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
    )
end

function clear!(film::Film)
    film.iteration_index[] = Int32(0)
    fill!(film.albedo, RGB{Float32}(0, 0, 0))
    fill!(film.normal, Vec3f(0, 0, 0))
    fill!(film.depth, 0.0f0)
end

# ============================================================================
# Auxiliary Buffer Filling (for denoising)
# ============================================================================

"""
    fill_aux_buffers!(film, scene, camera; has_infinite_lights=false)

Fill auxiliary buffers (albedo, normal, depth) by tracing primary rays.
These are used by `denoise!` for edge-aware filtering.
"""
function fill_aux_buffers!(film::Film, scene, camera; has_infinite_lights::Bool=false)
    albedo = film.albedo
    normal = film.normal
    depth = film.depth
    resolution = film.resolution
    crop_bounds = film.crop_bounds
    miss_depth = has_infinite_lights ? Float32(1e30) : Inf32

    backend = KA.get_backend(albedo)
    # Adapt the acceleration structure for kernel dispatch (TLAS -> StaticTLAS)
    accel = Adapt.adapt(backend, scene.accel)
    kernel! = aux_buffer_kernel!(backend)
    kernel!(
        albedo, normal, depth,
        resolution, crop_bounds,
        accel, camera, miss_depth;
        ndrange = length(albedo)
    )
    KA.synchronize(backend)
    return film
end

@kernel inbounds=true function aux_buffer_kernel!(albedo, normal, depth, resolution, crop_bounds, accel, camera, miss_depth::Float32)
    idx = @index(Global)

    h, _ = size(albedo)
    row = ((idx - 1) % h) + 1
    col = ((idx - 1) ÷ h) + 1

    px = Float32(col) + crop_bounds.p_min[1] - 1f0
    py = Float32(row) + crop_bounds.p_min[2] - 1f0
    pixel = Point2f(px + 0.5f0, py + 0.5f0)

    camera_sample = CameraSample(pixel, Point2f(0.5f0), 0f0)
    ray, ω = generate_ray(camera, camera_sample)

     if ω > 0f0
        hit, _primitive, si = intersect!(accel, ray)
        if hit
            normal[idx] = Vec3f(si.core.n)
            cam_pos = ray.o
            hit_pos = si.core.p
            depth[idx] = sqrt(sum((hit_pos .- cam_pos) .^ 2))
            albedo[idx] = RGB{Float32}(0.8f0, 0.8f0, 0.8f0)
        else
            normal[idx] = Vec3f(0f0, 0f0, 0f0)
            depth[idx] = miss_depth
            albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
        end
    else
        normal[idx] = Vec3f(0f0, 0f0, 0f0)
        depth[idx] = Inf32
        albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
    end
end

# ============================================================================
# Resource Cleanup
# ============================================================================

"""
    free!(film::Film)

Release GPU memory held by the film.  Does **not** synchronize.

**Precondition (caller's responsibility):** the GPU must be idle before
this is called.  The simplest way is to call this only after a
`colorbuffer` has completed (it issues `device_wait_idle`) or after
`sync!(scene)`.  Calling while a render is in flight is a use-after-free.
"""
function free!(film::Film)
    finalize(film.filter_table)
    finalize(film.framebuffer)
    finalize(film.albedo)
    finalize(film.normal)
    finalize(film.depth)
    finalize(film.postprocess)
    if film.aux_rays[] !== nothing
        finalize(film.aux_rays[])
        film.aux_rays[] = nothing
    end
    if film.aux_results[] !== nothing
        finalize(film.aux_results[])
        film.aux_results[] = nothing
    end
    return nothing
end
