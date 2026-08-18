# VolPath Integrator - Volumetric Path Tracing with Participating Media
# Wavefront implementation following pbrt-v4's VolPathIntegrator

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA

# How often the bounce loop synchronizes to check whether all rays have died
# (see the comment on the loop in `render!`). Smaller = exit closer to the
# true ray-death depth but more pipeline drains; 8 keeps the drain count at
# (live_rounds / 8) per sample while bounding the dead-round overshoot to 7
# rounds. Live-ray count is monotonic non-increasing, so the check is exact
# once it fires.
const EXIT_CHECK_INTERVAL = Int32(8)

# Benchmark A/B toggle for the bounce-loop early exit. Costs ~0.1s on scenes
# whose rays survive to max_depth (crown: gem media + RR keep a tail alive);
# saves 0.2-0.5s wherever rays die early (killeroo, materials).
const EARLY_EXIT_ENABLED = Ref(true)

# ============================================================================
# VolPath Integrator
# ============================================================================

"""
    VolPath <: Integrator

Volumetric path tracer using hero wavelength sampling and wavefront architecture.

Extends path tracing with support for participating media (fog, smoke, clouds).
Uses delta tracking with null scattering for efficient medium traversal.

The main loop follows pbrt-v4's structure:
1. If ray is in medium: run delta tracking
   - Absorption → terminate
   - Real scatter → sample direct light + phase function → continue
   - Null scatter → continue tracking
   - Escape medium → proceed to intersection
2. Intersect with scene
3. Handle surface (emission + BSDF) or environment light
"""
mutable struct VolPath <: Integrator
    max_depth::Int32
    samples_per_pixel::Int32
    russian_roulette_depth::Int32
    regularize::Bool  # Apply BSDF regularization after first non-specular bounce
    max_component_value::Float32  # Firefly suppression: clamp RGB components (pbrt-v4 style)
    filter_params::GPUFilterParams  # Filter for pixel reconstruction (pbrt-v4 style)
    filter_sampler_data::Union{Nothing, GPUFilterSamplerData}  # Tabulated data for importance sampling
    accumulation_eltype::DataType  # Element type for RGB/weight accumulators (Float64 or Float32)

    # Hardware ray tracing flag — when true, RayMakie creates HWTLAS instead of software TLAS
    hw_accel::Bool

    # Pixel sensor: spectral → RGB conversion with ISO, white balance, exposure
    sensor::PixelSensor

    # Cached render state
    state::Union{Nothing, VolPathState}

    # Cached initial medium detection (avoids per-sample GPU alloc + flush).
    # `initial_medium_camera_pos === nothing` means invalid.
    initial_medium_camera_pos::Any  # Point3f or nothing
    initial_medium_key::Any         # SetKey or nothing

    # Cached GPU-adapted filter sampler data (avoids re-uploading every sample).
    # nothing = not yet adapted.
    filter_sampler_gpu::Any

    # Cull mask passed to trace_closest_hits! / trace_closest_hits_indirect!.
    # ANDed against each TLAS instance's instance_mask; only instances where
    # (cull_mask & instance_mask) != 0 are visible to rays from this integrator.
    # Default 0xFF matches all instances (identical to old hard-coded behavior).
    cull_mask::UInt32
end

"""
    VolPath(; max_depth=8, samples=64, russian_roulette_depth=3, regularize=true,
            max_component_value=10, filter=GaussianFilter(),
            accumulation_eltype=Float32, hw_accel=nothing)

Create a VolPath integrator for volumetric path tracing.

# Arguments
- `max_depth`: Maximum path depth
- `samples`: Number of samples per pixel
- `russian_roulette_depth`: Depth at which to start Russian roulette
- `regularize`: Apply BSDF regularization to reduce fireflies (default: true).
  When enabled, near-specular BSDFs are roughened after the first non-specular
  bounce, following pbrt-v4's approach.
- `max_component_value`: Maximum RGB component value before clamping (default: 10).
  When set to a finite value, RGB values are scaled down if any component exceeds this.
  This is pbrt-v4's firefly suppression mechanism. Try values like 10.0 or 100.0.
- `filter`: Pixel reconstruction filter (default: GaussianFilter with radius 1.5, sigma 0.5).
  Supports BoxFilter, TriangleFilter, GaussianFilter, MitchellFilter, LanczosSincFilter.
  All filters use importance sampling with weight≈1 (pbrt-v4 compatible).
- `accumulation_eltype`: Element type for RGB/weight accumulators (default: Float32).
  Use Float64 on backends that support double precision for higher accumulation precision.
- `hw_accel`: Enable hardware ray tracing (default: false).
  When true and using LavaBackend, RayMakie creates an HWTLAS for hardware BVH traversal
  instead of software BVH. Dispatch happens on the accel type, not this flag.

Sensor simulation (ISO, exposure_time, white_balance) is applied during rendering
via the PixelSensor on the integrator state, matching pbrt-v4. The framebuffer
contains sensor-calibrated linear sRGB. Postprocessing is purely display mapping
(exposure, tonemapping, gamma).
"""
function VolPath(;
    max_depth::Int = 8,
    samples::Int = 64,
    russian_roulette_depth::Int = 3,
    regularize::Bool = true,
    max_component_value::Real = 10f0,
    filter::AbstractFilter = GaussianFilter(),  # pbrt-v4 default: Gaussian(1.5, 0.5)
    accumulation_eltype::DataType = Float32,
    hw_accel::Bool = false,
    sensor::PixelSensor = PixelSensor(),
    cull_mask::UInt32 = UInt32(0xFF)
)
    @assert accumulation_eltype in (Float32, Float64) "accumulation_eltype must be Float32 or Float64"
    # Build filter sampler data for importance sampling (nothing for Box/Triangle)
    sampler_data = GPUFilterSamplerData(filter)
    vp = VolPath(
        Int32(max_depth),
        Int32(samples),
        Int32(russian_roulette_depth),
        regularize,
        Float32(max_component_value),
        GPUFilterParams(filter),
        sampler_data,
        accumulation_eltype,
        hw_accel,
        sensor,
        nothing,           # state
        nothing,           # initial_medium_camera_pos
        nothing,           # initial_medium_key
        nothing,           # filter_sampler_gpu
        cull_mask,
    )
    # No GC finalizer — the VolPathState is shared with scene_state.integrator_state,
    # so proactive free! from a finalizer would free arrays still in use by the render loop.
    # GPU memory is freed by: explicit close(screen), or LavaArray's own DataRef finalizers.
    return vp
end

"""
    Base.close(vp::VolPath)

Release all GPU memory held by the integrator's cached render state and adapted scene.
"""
function Base.close(vp::VolPath)
    # Proactively free the GPU buffers backing vp.state — bare `vp.state = nothing`
    # relies on Julia GC to run finalizers, which races the next render's
    # allocation (peak memory becomes 2× state and can OOM on big scenes).
    # Symmetric with the integrator-change branch in RayMakie's
    # apply_screen_config!.  Same precondition: caller ensures GPU is idle.
    if vp.state !== nothing
        free!(vp.state)
    end
    vp.filter_sampler_gpu = nothing
    vp.state = nothing
    vp.initial_medium_camera_pos = nothing
    vp.initial_medium_key = nothing
    return nothing
end

# Dispatch wrapper: pass `vp` so external packages (e.g. hikari_integration.jl)
# can overload based on accel type.
function vp_trace_shadow_rays!(state::VolPathState, accel, media_interfaces, media, materials, ::VolPath)
    vp_trace_shadow_rays!(state, accel, media_interfaces, media, materials)
end

# Scene adaptation dispatch — overridden for HWAdaptedAccel in hikari_integration.jl
adapt_scene_for_render(backend, scene, ::VolPath) = Adapt.adapt(backend, scene)

# Camera medium detection dispatch — overridden for HWAdaptedAccel in hikari_integration.jl
detect_initial_medium(backend, accel, media_interfaces, camera_pos, ::VolPath) =
    detect_camera_medium(backend, accel, media_interfaces, camera_pos)

"""
    shades_surfaces_inline(accel) -> Bool

Does tracing already shade the surfaces it hits? The hardware RT path runs
per-material closest-hit shaders, so a hit is fully shaded by the time the
trace returns; the software BVH only records hits and leaves shading to the
post-hoc kernels. Only true for rays that never enter a medium — see
`chit_owns_surface` in `render!`.

The `HWAdaptedAccel` method lives in `hw-rt.jl`, with the other overrides on
the accel axis.
"""
shades_surfaces_inline(accel) = false

"""
    clear!(integrator::VolPath)

Clear the integrator's internal state (RGB and weight accumulators) for restarting progressive rendering.
"""
function Hikari.clear!(vp::VolPath)
    if vp.state !== nothing
        KernelAbstractions.fill!(vp.state.pixel_rgb, zero(eltype(vp.state.pixel_rgb)))
        KernelAbstractions.fill!(vp.state.pixel_weight_sum, zero(eltype(vp.state.pixel_weight_sum)))
    end
end

# ============================================================================
# Camera Ray Generation
# ============================================================================

"""
    vp_generate_camera_rays_kernel!(...)

Generate camera rays with per-pixel wavelength sampling and filter weight computation.
Following pbrt-v4's GetCameraSample: samples the filter, computes offset and weight.
"""
@kernel inbounds=true function vp_generate_camera_rays_kernel!(
    ray_queue,  # WorkQueue passed via Adapt
    wavelengths_per_pixel,
    pdf_per_pixel,
    filter_weight_per_pixel,
    @Const(height::Int32),
    @Const(camera),
    @Const(camera_needs_time::Bool),  # camera.shutter_open != shutter_close
    @Const(camera_needs_lens::Bool),  # camera.lens_radius > 0
    @Const(sample_idx::Int32),
    @Const(initial_medium_idx),
    @Const(filter_params::GPUFilterParams),
    filter_sampler_data,  # GPUFilterSamplerData or nothing for Box/Triangle
    rng  # SobolRNG passed via Adapt
)
    idx = @index(Global)
    num_pixels = rng.width * height

    @inbounds if idx <= num_pixels
        pixel_idx = idx - Int32(1)
        x = u_int32(mod(pixel_idx, rng.width)) + Int32(1)
        y = u_int32(div(pixel_idx, rng.width)) + Int32(1)

        # ZSobol sampling: deterministic low-discrepancy samples based on (pixel, sample_index)
        # Matches PBRT-v4's ZSobolSampler for better convergence.  Skip the
        # lens 2D sample for pinhole cameras and the time 1D sample for
        # shutter-stopped cameras — both are pure waste on those configs
        # and the camera-ray kernel is Sobol-bound.  Effects flags come
        # from the kernel arguments (computed CPU-side) so the GPU code
        # path is just two uniform branches, not a getproperty chain.
        pixel_sample = compute_pixel_sample(rng, Int32(x), Int32(y), sample_idx,
                                            camera_needs_time, camera_needs_lens)
        wavelength_u = pixel_sample.wavelength_u

        # Sample the filter to get offset and weight (pbrt-v4 style)
        # The filter sample uses the pixel jitter values as the 2D random input
        filter_u = Point2f(pixel_sample.jitter_x, pixel_sample.jitter_y)
        fs = filter_sample(filter_params, filter_sampler_data, filter_u)

        # Store filter weight for this sample (used in accumulation)
        filter_weight_per_pixel[idx] = fs.weight

        # Sample wavelengths for this pixel
        lambda = sample_wavelengths_visible(wavelength_u)

        # Store wavelengths and PDFs
        base = pixel_idx * Int32(4)
        wavelengths_per_pixel[base + Int32(1)] = lambda.lambda[1]
        wavelengths_per_pixel[base + Int32(2)] = lambda.lambda[2]
        wavelengths_per_pixel[base + Int32(3)] = lambda.lambda[3]
        wavelengths_per_pixel[base + Int32(4)] = lambda.lambda[4]
        pdf_per_pixel[base + Int32(1)] = lambda.pdf[1]
        pdf_per_pixel[base + Int32(2)] = lambda.pdf[2]
        pdf_per_pixel[base + Int32(3)] = lambda.pdf[3]
        pdf_per_pixel[base + Int32(4)] = lambda.pdf[4]

        # Film position with filter offset (flip Y for camera convention)
        # pbrt-v4: cs.pFilm = pPixel + fs.p + Vector2f(0.5f, 0.5f)
        # x,y are 1-based; convert to 0-based pixel center: (x-1)+0.5 = x-0.5
        p_film = Point2f(
            Float32(x) - 0.5f0 + fs.p[1],
            Float32(height) - Float32(y) + 0.5f0 + fs.p[2]
        )

        camera_sample = CameraSample(p_film, Point2f(pixel_sample.lens_u, pixel_sample.lens_v), pixel_sample.time)
        ray, weight = generate_ray(camera, camera_sample)

        if weight > 0f0
            raycore_ray = Raycore.Ray(o=ray.o, d=ray.d, t_max=ray.t_max)

            work_item = VPRayWorkItem(
                raycore_ray,
                Int32(0),  # depth
                lambda,    # Already a SampledWavelengths from sample_wavelengths_visible
                Int32(idx),
                SpectralRadiance(1f0),  # beta
                SpectralRadiance(1f0),  # r_u
                SpectralRadiance(1f0),  # r_l
                Point3f(0f0),           # prev_intr_p
                Vec3f(0f0, 0f0, 1f0),   # prev_intr_n
                1f0,                    # eta_scale
                false,                  # specular_bounce
                false,                  # any_non_specular_bounces
                initial_medium_idx
            )

            push!(ray_queue, work_item)
        end
    end
end

# ============================================================================
# Inline Sobol samples
# ============================================================================
#
# Previously, a separate `vp_generate_ray_samples_kernel!` ran once per bounce
# to pre-fill five `pixel_samples_*` SOA buffers (direct_uc, direct_u,
# indirect_uc, indirect_u, indirect_rr) from Sobol.  Per-kernel GPU timing
# (Lava.with_dispatch_timing, 2026-06-03) showed that kernel was 58 % of GPU
# time on killeroo native HW RT — the actual path-tracing kernels were
# 0.1 %.  The fix: each consumer (surface_direct_lighting_inner!,
# evaluate_material_inner!, medium_direct_lighting_inner!,
# medium_scatter_inner!) now generates only the Sobol dimensions it needs,
# inline from `(work.pixel_index, sample_idx, work.depth)`.  Eliminates the
# kernel dispatch + barrier per bounce, plus the SOA write of ~13 floats per
# pixel and the SOA read of those floats in the next dispatch (~150 MB of
# memory traffic per bounce on a 1.4 Mpx scene).


# ============================================================================
# Film Accumulation
# ============================================================================

"""
    vp_accumulate_to_rgb_kernel!(...)

Accumulate spectral radiance to RGB with filter weight support.
Following pbrt-v4's film accumulation pattern:
- Convert spectral L to XYZ using CIE matching functions
- Convert XYZ to linear sRGB
- Apply maxComponentValue clamping for firefly suppression
- Accumulate weighted RGB: rgbSum += weight * rgb
- Accumulate weight: weightSum += weight

Sensor imaging_ratio is applied here during spectral-to-RGB conversion, matching
pbrt-v4's PixelSensor::ToSensorRGB. White balance is handled via the output_matrix.
"""
@kernel inbounds=true function vp_accumulate_to_rgb_kernel!(
    pixel_rgb,
    pixel_weight_sum,
    @Const(pixel_L),
    @Const(wavelengths_per_pixel),
    @Const(pdf_per_pixel),
    @Const(filter_weight_per_pixel),
    @Const(cie_x), @Const(cie_y), @Const(cie_z),
    @Const(n_pixels::Int32),
    @Const(max_component_value::Float32),
    @Const(output_matrix::Mat3f),
    @Const(imaging_ratio::Float32),
)
    pixel_idx = @index(Global)

    # Reconstruct CIE table from passed arrays
    cie_table = CIEXYZTable(cie_x, cie_y, cie_z)

    @inbounds if pixel_idx <= n_pixels
        base = (pixel_idx - Int32(1)) * Int32(4)

        L = load(pixel_L, base + Int32(1), SpectralRadiance)

        lambda_tuple = load(wavelengths_per_pixel, base + Int32(1), NTuple{4, Float32})
        pdf_tuple = load(pdf_per_pixel, base + Int32(1), NTuple{4, Float32})

        lambda = Wavelengths(lambda_tuple, pdf_tuple)

        # Convert spectral to response (CIE XYZ or sensor RGB)
        response = spectral_to_xyz(cie_table, L, lambda)
        sensor_rgb = max.(0f0, response) * imaging_ratio

        # Apply maxComponentValue clamping in sensor space (pbrt-v4: before output matrix)
        m = maximum(sensor_rgb)
        if m > max_component_value
            sensor_rgb = sensor_rgb * (max_component_value / m)
        end

        # Convert sensor RGB → output sRGB
        rgb = output_matrix * sensor_rgb

        # Get filter weight for this sample (pbrt-v4 style weighted accumulation)
        weight = filter_weight_per_pixel[pixel_idx]

        # Accumulate weighted RGB (pbrt-v4: pixel.rgbSum[c] += weight * rgb[c])
        rgb_base = (pixel_idx - Int32(1)) * Int32(3)
        Atomix.@atomic pixel_rgb[rgb_base + Int32(1)] += weight * rgb[1]
        Atomix.@atomic pixel_rgb[rgb_base + Int32(2)] += weight * rgb[2]
        Atomix.@atomic pixel_rgb[rgb_base + Int32(3)] += weight * rgb[3]

        # Accumulate weight (pbrt-v4: pixel.weightSum += weight)
        Atomix.@atomic pixel_weight_sum[pixel_idx] += weight
    end
end

"""
    vp_finalize_film_kernel!(...)

Finalize film by dividing weighted RGB sum by weight sum.
Following pbrt-v4's RGBFilm::GetPixelRGB:
- rgb = rgbSum / weightSum (if weightSum != 0)
"""
@kernel inbounds=true function vp_finalize_film_kernel!(
    framebuffer,
    @Const(pixel_rgb),
    @Const(pixel_weight_sum),
    @Const(width::Int32), @Const(height::Int32)
)
    pixel_idx = @index(Global)

    @inbounds if pixel_idx <= width * height
        px = ((pixel_idx - Int32(1)) % width) + Int32(1)
        py = ((pixel_idx - Int32(1)) ÷ width) + Int32(1)

        rgb_base = (pixel_idx - Int32(1)) * Int32(3)
        weight_sum = pixel_weight_sum[pixel_idx]

        # Normalize by weight sum (pbrt-v4 style).
        # Use zero/one relative to weight_sum's type so the kernel stays in Float32
        # when accumulation_eltype=Float32 (the GPU-compatible default).
        if weight_sum > zero(weight_sum)
            inv_weight = one(weight_sum) / weight_sum
            r = Float32(pixel_rgb[rgb_base + Int32(1)] * inv_weight)
            g = Float32(pixel_rgb[rgb_base + Int32(2)] * inv_weight)
            b = Float32(pixel_rgb[rgb_base + Int32(3)] * inv_weight)
        else
            r = 0f0
            g = 0f0
            b = 0f0
        end

        # Store linear HDR values (no clamping, no gamma)
        # Postprocessing will handle tonemapping and gamma correction
        framebuffer[py, px] = RGB{Float32}(r, g, b)
    end
end

# ============================================================================
# Main Render Function
# ============================================================================

"""
    finalize_film!(vp::VolPath, film::Film)

Run the per-pixel `pixel_rgb / pixel_weight_sum` divide and write the result
to `film.framebuffer`.  Called automatically at the end of `render!` unless
`finalize_framebuffer=false` is passed (used by batched callers that finalize
once after all samples).
"""
function finalize_film!(vp::VolPath, film::Film)
    state = vp.state
    state === nothing && return nothing
    img = film.framebuffer
    height, width = size(img)
    backend = KA.get_backend(img)
    n_pixels = width * height
    kernel! = vp_finalize_film_kernel!(backend)
    kernel!(
        img, state.pixel_rgb, state.pixel_weight_sum,
        Int32(width), Int32(height);
        ndrange=Int(n_pixels),
    )
    return nothing
end

"""
    render!(vp::VolPath, scene, film, camera)

Render one iteration/sample using volumetric spectral wavefront path tracing.

This function is allocation-free and renders a single sample, accumulating
results in the film. The iteration index is read from and incremented in
`film.iteration_index`.

For progressive rendering, call this repeatedly. For complete rendering,
use the main call function `(vp::VolPath)(scene, film, camera)` which wraps
this in a loop.
"""
function render!(
    vp::VolPath,
    scene::AbstractScene,
    film::Film,
    camera::Camera;
    # When `false`, skip the per-sample `vp_finalize_film_kernel!` dispatch
    # (which divides accumulated RGB by weight and writes the framebuffer).
    # The caller must run `finalize_film!(vp, film)` itself after the last
    # sample.  For batched offline rendering (RayMakie's `colorbuffer` loop)
    # the intermediate framebuffers are never observed, so skipping all but
    # the last call saves one full-resolution dispatch per sample —
    # measured ~17 % of captured GPU time on killeroo 32 spp.  Defaults to
    # `true` so live-preview callers (Makie's renderloop) keep getting
    # per-sample updates.
    finalize_framebuffer::Bool = true,
)
    img = film.framebuffer
    height, width = size(img)
    backend = KA.get_backend(img)

    # Adapt scene for kernel dispatch (TLAS → StaticTLAS, MultiTypeSet → StaticMultiTypeSet).
    # Cheap: `Adapt.adapt(backend, scene.accel)` reads `scene.accel.static_tlas`
    # after a no-op `sync!`. Do NOT cache this return across calls — consumers
    # MUST re-read per dispatch so scene mutations (push!/delete!) are visible.
    adapted = adapt_scene_for_render(backend, scene, vp)
    accel = adapted.accel
    materials = adapted.materials
    media = adapted.media
    media_interfaces = adapted.media_interfaces
    lights = adapted.lights

    # Allocate or validate state
    # Note: Rebuild state if lights changed (num_lights mismatch) to update light sampler
    n_lights = length(lights)
    # `hit_area_light_queue` and the per-material typed queues are vestigial
    # ONLY when the HW per-material chit fully owns the surface shading.  On
    # medium-bearing scenes the wavefront medium-survives-to-surface path
    # (`delta-tracking.jl`'s `vp_sample_medium_kernel!`) still uses
    # `enqueue_after_intersection!` to push hits into both queues, so they
    # must be full-capacity even on HW.
    chit_owns_surface = shades_surfaces_inline(accel) && isempty(media)

    has_media = !isempty(media)
    if vp.state === nothing ||
       vp.state.width != width ||
       vp.state.height != height ||
       vp.state.num_lights != n_lights ||
       # Queues allocated as capacity-1 placeholders (no-media medium queues,
       # chit-owns-surface hit/area-light queues) must grow if a scene that
       # needs them shows up on a reused integrator.
       (has_media && Int(vp.state.medium_sample_queue.capacity) < width * height) ||
       (!chit_owns_surface && Int(vp.state.hit_surface_queue.capacity) < width * height)
        # Free the previous state's GPU buffers before reallocating — a bare
        # reassign races GC finalizers and doubles peak memory.
        if vp.state !== nothing
            KA.synchronize(backend)
            free!(vp.state)
        end
        # Use original scene.lights (MultiTypeSet) for PowerLightSampler (needs .backend)
        # Ensure SobolRNG has enough bits for progressive rendering:
        # When samples_per_pixel=1 (interactive mode), log2_spp=0 causes sample_idx
        # bits to collide with pixel morton code bits, making every sample identical.
        # Use at least 4096 so log2_spp=12, supporting up to 4096 progressive samples.
        sobol_spp = max(Int(vp.samples_per_pixel), 4096)
        vp.state = VolPathState(backend, width, height, scene.lights;
                                max_depth=vp.max_depth,
                                rr_depth=vp.russian_roulette_depth,
                                scene_radius=world_radius(scene),
                                samples_per_pixel=sobol_spp,
                                sampler_seed=UInt32(0),
                                accumulation_eltype=vp.accumulation_eltype,
                                sensor=vp.sensor,
                                hw_accel=chit_owns_surface,
                                has_media=has_media)
    end
    state = vp.state

    n_pixels = width * height

    # Build (or refresh) the per-material typed queues from the adapted
    # materials set.  On the HW per-material chit path the chit shaders fully
    # shade hits inline and the typed queues are never drained, so we use a
    # capacity=1 placeholder.  SW BVH and HW-with-media still route surface
    # hits through these to avoid the monolithic with_index BSDF switch.
    pmq_capacity = chit_owns_surface ? 1 : n_pixels
    ensure_per_material_queue!(state, materials, pmq_capacity)

    # Get current iteration index and increment
    sample_idx = film.iteration_index[] + Int32(1)
    film.iteration_index[] = sample_idx

    # Get SobolRNG from state (allocated once)
    sobol_rng = state.sobol_rng

    # Get accumulators from state (allocation-free)
    pixel_rgb = state.pixel_rgb
    pixel_weight_sum = state.pixel_weight_sum
    wavelengths_per_pixel = state.wavelengths_per_pixel
    pdf_per_pixel = state.pdf_per_pixel
    filter_weight_per_pixel = state.filter_weight_per_pixel

    # Detect which medium the camera is inside (vacuum if outside all media)
    # Cached: camera position doesn't change between samples, so detect once per render
    camera_pos = get_camera_position(camera)
    if vp.initial_medium_camera_pos === camera_pos && vp.initial_medium_key !== nothing
        initial_medium = vp.initial_medium_key
    else
        initial_medium = detect_initial_medium(backend, accel, media_interfaces, camera_pos, vp)
        vp.initial_medium_camera_pos = camera_pos
        vp.initial_medium_key = initial_medium
    end

    # Clear spectral buffer (pixel_L) for this sample iteration + reset the
    # ray queue.  Both are fills; the GPU's per-dispatch barrier between
    # them is wasted (they touch disjoint memory).  Wrap in a
    # `concurrent_dispatch_group` so they overlap.
    concurrent_dispatch_group() do
        reset_film!(state)
        empty!(current_ray_queue(state))
    end

    # Generate camera rays with filter sampling (pbrt-v4 style) and ZSobol sampler
    # Adapt filter sampler data to GPU — cache on struct to avoid re-uploading every sample
    if vp.filter_sampler_gpu === nothing
        vp.filter_sampler_gpu = Adapt.adapt(backend, vp.filter_sampler_data)
    end
    filter_sampler_data_gpu = vp.filter_sampler_gpu

    # Compute camera effects flags CPU-side so the kernel doesn't have to
    # walk getproperty chains into Camera.core.shutter_*.  These are uniform
    # across all pixels of the render.
    camera_needs_time = camera_uses_motion_blur(camera)
    camera_needs_lens = camera_uses_lens(camera)
    kernel! = vp_generate_camera_rays_kernel!(backend)
    kernel!(
        current_ray_queue(state),  # WorkQueue passed via Adapt
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        Int32(height),
        camera, camera_needs_time, camera_needs_lens,
        sample_idx, initial_medium, vp.filter_params,
        filter_sampler_data_gpu,
        sobol_rng;
        ndrange=Int(n_pixels)
    )

    # Path tracing loop - following pbrt-v4 wavefront architecture
    # All inner kernels use indirect dispatch (0 rays = GPU no-op), so we do
    # NOT check queue sizes every bounce — that forces a flush + fence per
    # round and kills GPU pipelining. But running ALL max_depth rounds
    # unconditionally is just as bad on deep scenes: rays die off long before
    # max_depth (Crown: maxdepth 100, ray population < 1% past depth ~30),
    # and every dead round still costs its dispatch commands + barriers —
    # a measured ~1.3 ms/round, ≈2.1 s of Crown's 3.6 s render.
    #
    # Compromise: every EXIT_CHECK_INTERVAL rounds, synchronize once and read
    # the input ray queue's 4-byte BAR counter. The live-ray count is
    # monotonically non-increasing across rounds (a ray either continues 1:1
    # — surface bounce, medium scatter, null crossing — or dies; nothing
    # splits), so "queue empty" is a stable exit condition. Worst case we
    # record EXIT_CHECK_INTERVAL-1 extra dead rounds past the true death
    # point, and pay (live_rounds / interval) pipeline drains per sample.
    for depth in 0:(vp.max_depth - 1)
        if EARLY_EXIT_ENABLED[] && depth > 0 && depth % EXIT_CHECK_INTERVAL == 0
            KA.synchronize(backend)
            length(current_ray_queue(state)) == 0 && break
        end
        # Sobol samples are generated inline inside each consumer kernel —
        # see the "Inline Sobol samples" block earlier in this file.  The
        # consumers compute their dimensions from (work.depth, sample_idx)
        # so we just thread `sample_idx` through the dispatchers.
        reset_iteration_queues!(state)

        # Trace + shade in one dispatch for non-medium rays. Medium rays still
        # take the old path: the fused kernel pushes them to medium_sample_queue
        # and the medium pipeline below drains it (and any medium-originated
        # surface escapes into the per-material queues, which `vp_shade_surfaces!`
        # drains after the medium kernels run). Surface-only scenes (the
        # common case) bypass `hit_surface_queue` entirely — saves one dispatch
        # + barrier per bounce plus ~170 MB of work-item materialization on a
        # 1.4M-pixel render.
        vp_trace_and_shade!(state, accel, media_interfaces, media, materials, lights,
                            sample_idx,
                            camera, Int32(vp.samples_per_pixel), vp.regularize)

        # Medium sampling — indirect dispatch handles empty queues (0 groups = no-op)
        if !isempty(media)
            vp_sample_medium_interaction!(state, media, materials)
        end

        # NOTE on grouping: only the per-material shading dispatches share a
        # deferred indirect group (inside `vp_shade_surfaces!`). Grouping the
        # medium DL/scatter + escaped + emitters stages in with the shading
        # was tried (2026-06-10) and BENCHMARKED WORSE on volume/trace-heavy
        # scenes (bunny SW +25%, killeroo SW +22%) even though it removes
        # barriers — co-scheduling the heavy ray-query kernels (medium DL's
        # shadow transmittance, the shading kernels' inline shadow traces)
        # thrashes cache/occupancy. Sequential pairs win there; the fused
        # multi-prepare group only pays off for the 12 SMALL per-type
        # shading dispatches (materials SW −25%, crown SW −5%).
        if !isempty(media)
            if length(lights) > 0
                vp_sample_medium_direct_lighting!(state, lights, sample_idx)
            end
            vp_sample_medium_scatter!(state, sample_idx)
        end

        # Escaped rays — lights check is CPU-side static scene data
        if length(lights) > 0
            vp_handle_escaped_rays!(state, lights)
        end

        # HW per-material chit path: emission MIS and per-material BSDF /
        # DL / RR / continuation already ran INSIDE each material's chit
        # shader during `vp_trace_and_shade!`. `hit_area_light_queue` and
        # `per_material_queue` are unused on this path, so we skip both
        # `vp_handle_emitters!` and `vp_shade_surfaces!`. The SW BVH path
        # still drains them — the post-hoc kernels are the only place
        # surface shading runs there. Medium-bearing HW scenes also still
        # need them: the medium-survives-to-surface path
        # (`vp_sample_medium_kernel!` in delta-tracking.jl) populates the
        # same queues via `enqueue_after_intersection!`.
        if !chit_owns_surface
            # Emission MIS — drains hit_area_light_queue. Direct port of
            # pbrt-v4's "Handle emitters hit by indirect rays" kernel
            # (wavefront/integrator.cpp:540).
            if length(lights) > 0
                vp_handle_emitters!(state, lights)
            end

            # Per-material shading — drains the typed index queues. Both
            # producers fill them: the surface trace for non-medium rays and
            # delta tracking's survive-to-surface path, through the same
            # `enqueue_after_intersection!`. One dispatch per concrete
            # material type; each kernel contains only one material's BSDF, so
            # SPIR-V stays small and the register cliff (128 regs/thread) is
            # avoided. The dispatches share a fused-prepare deferred group.
            vp_shade_surfaces!(state, accel, media_interfaces, media,
                               materials, lights,
                               sample_idx,
                               camera,
                               Int32(vp.samples_per_pixel), vp.regularize)
        end

        vp_trace_shadow_rays!(state, accel, media_interfaces, media, materials, vp)

        swap_ray_queues!(state)
    end

    # Accumulate this sample's spectral radiance to RGB with filter weights (pbrt-v4 style)
    # Sensor imaging_ratio and output_matrix are applied here, matching pbrt-v4's PixelSensor.
    kernel! = vp_accumulate_to_rgb_kernel!(backend)
    kernel!(
        pixel_rgb, pixel_weight_sum, state.pixel_L,
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        state.cie_table.cie_x, state.cie_table.cie_y, state.cie_table.cie_z,
        Int32(n_pixels),
        vp.max_component_value,
        state.output_matrix,
        state.imaging_ratio;
        ndrange=Int(n_pixels)
    )

    # Update film: divide weighted sum by weight sum (pbrt-v4 style).
    # The kwarg lets batched callers (RayMakie's colorbuffer loop) skip the
    # intermediate framebuffer writes between samples and only finalize once
    # at the end — saves a 1.4 Mpx dispatch per sample on killeroo.
    if finalize_framebuffer
        finalize_film!(vp, film)
    end

    return nothing
end

"""
    (vp::VolPath)(scene, film, camera)

Render using volumetric spectral wavefront path tracing.

Media are automatically extracted from `MediumInterface` materials during scene
construction and stored in `scene.media`.

The render loop follows pbrt-v4's VolPathIntegrator:
1. Generate camera rays (possibly in a medium)
2. For each bounce:
   a. Delta tracking for rays in media → scatter or pass through
   b. Trace rays to find intersections
   c. Handle escaped rays (environment light)
   d. Handle surface hits (emission + direct lighting + BSDF sampling)
3. Accumulate spectral samples to RGB
"""
function (vp::VolPath)(
    scene::AbstractScene,
    film::Film,
    camera::Camera
)
    # Reset iteration counter for fresh render
    film.iteration_index[] = Int32(0)
    # Clear RGB and weight accumulators for fresh render
    clear!(vp)
    # Render all samples by calling render! repeatedly
    # Flush GPU between samples to avoid driver timeout on long renders.
    # Each sample generates ~100 dispatches; without flushing, N samples
    # accumulate into a single massive GPU submission that can exceed
    # the driver's preemption timeout (DEVICE_LOST).
    backend = KA.get_backend(film.framebuffer)
    for _ in 1:vp.samples_per_pixel
        render!(vp, scene, film, camera)
        KA.synchronize(backend)
    end

    postprocess!(film)
    return film.postprocess
end
