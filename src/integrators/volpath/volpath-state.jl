# VolPath state container
# Uses the unified WorkQueue from integrators/workqueue.jl

import KernelAbstractions as KA

# ============================================================================
# SOA Override for VolPath Work Items
# ============================================================================

# Override should_use_soa for VolPath work items that benefit from SOA layout.
#
# pbrt-v4's wavefront integrator declares its work items in a .soa DSL
# (wavefront/workitems.soa) that the build system expands into per-field
# SOA<T> templates. The reason: every queue is large enough (≥ scene-pixel-
# count entries) and each shading kernel reads many fields out of each work
# item across many threads — the AOS stride wastes most of the memory
# bandwidth on cache lines whose other fields the warp doesn't need this
# pass. SOA fixes this by giving each field its own coalesced array.
#
# The big items (≥ 64 B) are the ones where the bandwidth difference shows
# up at render scale. The defaults below cover every queue that's heavy
# enough to matter: the two ray queues, surface-hit, material-eval, shadow,
# escaped, and the two medium queues. Trait dispatch is read by
# `WorkQueue{T}(backend, capacity)` which defaults `soa=should_use_soa(T)` —
# so flagging a new work-item type here is sufficient to opt it in.
should_use_soa(::Type{VPRayWorkItem}) = true
should_use_soa(::Type{VPShadowRayWorkItem}) = true
should_use_soa(::Type{VPEscapedRayWorkItem}) = true
should_use_soa(::Type{VPHitSurfaceWorkItem}) = true
should_use_soa(::Type{VPMaterialEvalWorkItem}) = true
should_use_soa(::Type{VPMediumSampleWorkItem}) = true
should_use_soa(::Type{VPMediumScatterWorkItem}) = true
# VPRaySamples used to be a per-pixel Sobol cache filled by
# vp_generate_ray_samples_kernel!; that kernel and its buffer are gone now
# (samples are computed inline in each consumer), so the SOA hint is no
# longer needed.

# ============================================================================
# VolPath State Container
# ============================================================================

"""
    VolPathState

Contains all work queues and buffers for VolPath wavefront rendering.
"""
mutable struct VolPathState{Backend}
    backend::Backend

    # Ray queues (double-buffered for iteration)
    ray_queue_a::WorkQueue{VPRayWorkItem}
    ray_queue_b::WorkQueue{VPRayWorkItem}
    current_ray_queue::Symbol  # :a or :b

    # Medium sample queue (rays in medium with bounded t_max from intersection)
    medium_sample_queue::WorkQueue{VPMediumSampleWorkItem}

    # Medium scatter queue (real scattering events from delta tracking)
    medium_scatter_queue::WorkQueue{VPMediumScatterWorkItem}

    # Surface hit queue (rays that hit surfaces and are NOT in medium)
    hit_surface_queue::WorkQueue{VPHitSurfaceWorkItem}

    # Material evaluation queue
    material_queue::WorkQueue{VPMaterialEvalWorkItem}

    # Shadow ray queue
    shadow_queue::WorkQueue{VPShadowRayWorkItem}

    # Escaped ray queue
    escaped_queue::WorkQueue{VPEscapedRayWorkItem}

    # Hit area-light queue — pbrt-v4 hitAreaLightQueue (wavefront/integrator.h).
    # Surface hits on emissive triangles are pushed here in PARALLEL to the
    # per-material queue push (see `enqueue_after_intersection!`); a dedicated
    # emitter kernel (`vp_handle_emitters!`) drains it between trace and
    # shade. Keeps the per-material kernels free of emission-MIS code +
    # eliminates the corresponding fields (`arealight_flat_idx`,
    # `triangle_area`, `t_hit`, `prev_intr_p`, `prev_intr_n`) from the path
    # that surface BSDF eval traverses.
    hit_area_light_queue::WorkQueue{VPHitAreaLightWorkItem}

    # Film buffer (spectral radiance per pixel, 4 wavelengths)
    pixel_L::AbstractVector{Float32}

    # Accumulators for progressive rendering (moved from main loop)
    # NOTE: Default Float64 matches pbrt-v4's double precision accumulators (film.h:304-305)
    # Use Float32 for OpenCL backends that don't support double precision atomics
    pixel_rgb::AbstractVector  # n_pixels * 3 (RGB accumulator)
    pixel_weight_sum::AbstractVector  # n_pixels (filter weight accumulator)
    wavelengths_per_pixel::AbstractVector{Float32}  # n_pixels * 4 (wavelength samples)
    pdf_per_pixel::AbstractVector{Float32}  # n_pixels * 4 (wavelength PDFs)
    filter_weight_per_pixel::AbstractVector{Float32}  # n_pixels (filter weight per sample)

    # RGB to spectrum table
    rgb2spec_table::RGBToSpectrumTable

    # Spectral response table (CIE XYZ or camera sensor curves)
    cie_table::CIEXYZTable

    # Sensor output: response → output RGB matrix + exposure scaling
    output_matrix::Mat3f
    imaging_ratio::Float32

    # BVH Light Sampler data (spatially-aware importance sampling)
    bvh_nodes::AbstractVector{LightBVHNode}           # BVH node array (GPU)
    light_to_bit_trail::AbstractVector{UInt32}         # Per-light bit trail (GPU)
    infinite_light_indices::AbstractVector{Int32}       # Flat indices of infinite lights (GPU)
    num_bvh_lights::Int32                               # Count of bounded lights in BVH
    num_infinite_lights::Int32                          # Count of infinite lights
    num_lights::Int32

    # Render parameters
    max_depth::Int32
    rr_depth::Int32
    width::Int32
    height::Int32

    # Sobol RNG for low-discrepancy sampling (allocated once, reused across frames)
    sobol_rng::Any  # SobolRNG or nothing

    # Hardware RT buffers (lazily allocated when using HW-accelerated tracing)
    hw_primary_ray_buf::Any      # RTRay buffer for primary rays
    hw_primary_result_buf::Any   # RTHitResult buffer for primary rays
    hw_shadow_states::Any        # ShadowIterState buffer
    hw_shadow_ray_buf::Any       # RTRay buffer for shadow rays
    hw_shadow_result_buf::Any    # RTHitResult buffer for shadow rays

    # Per-material-type shading queues (one `WorkQueue{TypedHit{T}}` per
    # concrete material type `T` in the scene).  Built lazily on first render
    # from the scene's adapted `StaticMultiTypeSet` of materials; rebuilt
    # when the type tuple changes (different scene).  Type is left `Any`
    # because the concrete `MultiTypeMaterialQueue{Qs}` signature depends on
    # which materials are present.
    per_material_queue::Any              # MultiTypeMaterialQueue{...} | nothing
    per_material_queue_signature::Any    # type tuple, used as freshness check
end

"""
    free!(state::VolPathState)

Release all GPU memory held by the VolPath render state (work queues,
pixel buffers, tables).  Does **not** synchronize.

**Precondition (caller's responsibility):** the GPU must be idle.  Same
rule as `free!(::Film)` — pair this with the end of a `colorbuffer` / a
`sync!(scene)`, or explicitly `KA.synchronize(backend)` before calling.
"""
function free!(state::VolPathState)
    # Work queues (bulk of GPU memory — each holds items + size arrays)
    free!(state.ray_queue_a)
    free!(state.ray_queue_b)
    free!(state.medium_sample_queue)
    free!(state.medium_scatter_queue)
    free!(state.hit_surface_queue)
    free!(state.material_queue)
    free!(state.shadow_queue)
    free!(state.escaped_queue)
    free!(state.hit_area_light_queue)
    state.per_material_queue !== nothing && free!(state.per_material_queue)

    # Pixel buffers
    finalize(state.pixel_L)
    finalize(state.pixel_rgb)
    finalize(state.pixel_weight_sum)
    finalize(state.wavelengths_per_pixel)
    finalize(state.pdf_per_pixel)
    finalize(state.filter_weight_per_pixel)

    # BVH light sampler data
    finalize(state.bvh_nodes)
    finalize(state.light_to_bit_trail)
    finalize(state.infinite_light_indices)

    # Sobol RNG state
    if state.sobol_rng !== nothing
        rng = state.sobol_rng
        for name in fieldnames(typeof(rng))
            arr = getfield(rng, name)
            if arr isa AbstractArray
                finalize(arr)
            end
        end
    end

    return nothing
end

function VolPathState(
    backend,
    width::Integer,
    height::Integer,
    lights::Raycore.MultiTypeSet;  # MultiTypeSet of lights (has backend for GPU allocation)
    max_depth::Integer = 8,
    rr_depth::Integer = 1,
    queue_capacity::Integer = width * height,
    scene_radius::Float32 = 10f0,  # Scene bounding sphere radius for light power estimation
    samples_per_pixel::Integer = 1,  # For SobolRNG parameter computation
    sampler_seed::UInt32 = UInt32(0),  # Scrambling seed for Sobol
    accumulation_eltype::DataType = Float32,  # Element type for accumulators (Float32 for OpenCL)
    sensor::PixelSensor = PixelSensor(),  # Pixel sensor for spectral → RGB conversion
    # When HW per-material chit slots own the shading, the post-hoc
    # `vp_handle_emitters!` and `vp_shade_typed!` kernels never run, so the
    # `hit_area_light_queue` and the typed per-material queues are unused.
    # `hw_accel=true` allocates them with capacity=1 (placeholder slots that
    # the chit body never touches) so the SoA queue allocations don't blow
    # past GPU memory on scenes with many material types (e.g. `materials`
    # scene at 12 types × 2 MP pixels was ~6 GB of vestigial buffers).
    hw_accel::Bool = false,
)
    n_pixels = width * height
    vestigial_capacity = hw_accel ? 1 : queue_capacity

    # Create work queues
    ray_queue_a = WorkQueue{VPRayWorkItem}(backend, queue_capacity)
    ray_queue_b = WorkQueue{VPRayWorkItem}(backend, queue_capacity)
    medium_sample_queue = WorkQueue{VPMediumSampleWorkItem}(backend, queue_capacity)
    medium_scatter_queue = WorkQueue{VPMediumScatterWorkItem}(backend, queue_capacity)
    hit_surface_queue = WorkQueue{VPHitSurfaceWorkItem}(backend, queue_capacity)
    material_queue = WorkQueue{VPMaterialEvalWorkItem}(backend, queue_capacity)
    shadow_queue = WorkQueue{VPShadowRayWorkItem}(backend, queue_capacity)
    escaped_queue = WorkQueue{VPEscapedRayWorkItem}(backend, queue_capacity)
    hit_area_light_queue = WorkQueue{VPHitAreaLightWorkItem}(backend, vestigial_capacity)

    # Film buffer (4 wavelengths per pixel)
    pixel_L = KA.allocate(backend, Float32, n_pixels * 4)
    KA.fill!(pixel_L, 0f0)

    # Accumulators for progressive rendering (configurable eltype for OpenCL compatibility)
    pixel_rgb = KA.allocate(backend, accumulation_eltype, n_pixels * 3)
    KA.fill!(pixel_rgb, zero(accumulation_eltype))
    pixel_weight_sum = KA.allocate(backend, accumulation_eltype, n_pixels)
    KA.fill!(pixel_weight_sum, zero(accumulation_eltype))
    wavelengths_per_pixel = KA.allocate(backend, Float32, n_pixels * 4)
    KA.fill!(wavelengths_per_pixel, 0f0)
    pdf_per_pixel = KA.allocate(backend, Float32, n_pixels * 4)
    KA.fill!(pdf_per_pixel, 0f0)
    filter_weight_per_pixel = KA.allocate(backend, Float32, n_pixels)
    KA.fill!(filter_weight_per_pixel, 0f0)

    # Load lookup tables to GPU (sensor determines response curves)
    rgb2spec_table = to_gpu(backend, get_srgb_table())
    cie_table = to_gpu(backend, sensor_response_table(sensor.sensor_name))

    # Build BVH light sampler (spatially-aware importance sampling)
    n_lights = length(lights)
    if n_lights > 0
        bvh_sampler = BVHLightSampler(lights; scene_radius=scene_radius)
        bvh_gpu = bvh_to_gpu(backend, bvh_sampler)
        bvh_nodes = bvh_gpu.nodes
        light_to_bit_trail = bvh_gpu.light_to_bit_trail
        infinite_light_indices = bvh_gpu.infinite_light_indices
        num_bvh_lights = bvh_gpu.num_bvh_lights
        num_infinite_lights = bvh_gpu.num_infinite_lights
    else
        bvh_nodes = KA.allocate(backend, LightBVHNode, 1)
        light_to_bit_trail = KA.allocate(backend, UInt32, 1)
        infinite_light_indices = KA.allocate(backend, Int32, 1)
        num_bvh_lights = Int32(0)
        num_infinite_lights = Int32(0)
    end

    # Create SobolRNG (allocated once, reused across frames)
    sobol_rng = SobolRNG(backend, sampler_seed, width, height, samples_per_pixel)

    return VolPathState(
        backend,
        ray_queue_a, ray_queue_b, :a,
        medium_sample_queue, medium_scatter_queue,
        hit_surface_queue, material_queue, shadow_queue, escaped_queue,
        hit_area_light_queue,
        pixel_L, pixel_rgb, pixel_weight_sum,
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        rgb2spec_table, cie_table,
        sensor.output_from_sensor, sensor.imaging_ratio,
        bvh_nodes, light_to_bit_trail, infinite_light_indices,
        num_bvh_lights, num_infinite_lights, Int32(n_lights),
        Int32(max_depth), Int32(rr_depth), Int32(width), Int32(height),
        sobol_rng,
        # HW RT buffers (lazily allocated)
        nothing, nothing, nothing, nothing, nothing,
        # Per-material typed queues (built lazily once we see the scene's
        # adapted materials).
        nothing, nothing,
    )
end

"""
    ensure_per_material_queue!(state, materials_static, capacity)

Lazily build the per-material typed queue from the scene's adapted
`StaticMultiTypeSet` of materials.  Rebuilt when the type signature
changes (different scene); a no-op cache hit otherwise.  Called once at
the start of each `render!` after scene adaptation.
"""
function ensure_per_material_queue!(state::VolPathState, materials_static, capacity::Integer)
    sig = typeof(materials_static).parameters[1]   # Data tuple of StaticMultiTypeSet
    if state.per_material_queue === nothing || state.per_material_queue_signature !== sig
        state.per_material_queue !== nothing && free!(state.per_material_queue)
        state.per_material_queue = build_per_material_queues(materials_static, capacity, state.backend)
        state.per_material_queue_signature = sig
    end
    return state.per_material_queue
end

# ============================================================================
# State Helpers
# ============================================================================

"""Get the current ray queue based on the queue selector."""
function current_ray_queue(state::VolPathState)
    state.current_ray_queue == :a ? state.ray_queue_a : state.ray_queue_b
end

"""Get the next ray queue (the one not currently active)."""
function next_ray_queue(state::VolPathState)
    state.current_ray_queue == :a ? state.ray_queue_b : state.ray_queue_a
end

"""Swap the ray queue selector."""
function swap_ray_queues!(state::VolPathState)
    state.current_ray_queue = state.current_ray_queue == :a ? :b : :a
end

"""Reset all processing queues for a new bounce.

The empties are tiny 1-element `fill!`s, but the GPU normally inserts a
full pipeline barrier between each dispatch.  On a typical render that's
~50 fills/sample, each costing ~600 µs of barrier-wait — a measured
~1 s/render of pure between-dispatch idle on killeroo.  The empties
write to disjoint queue size counters and have no inter-dependencies, so
`concurrent_dispatch_group` lets them all overlap on the GPU with just
one pre-barrier against the prior bounce's reads."""
function reset_processing_queues!(state::VolPathState)
    concurrent_dispatch_group() do
        empty!(state.medium_sample_queue)
        empty!(state.medium_scatter_queue)
        empty!(state.hit_surface_queue)
        empty!(state.material_queue)
        empty!(state.shadow_queue)
        empty!(state.escaped_queue)
        empty!(state.hit_area_light_queue)
        empty!(next_ray_queue(state))
    end
end

"""Reset iteration queues before ray tracing.  Same barrier-collapsing
trick as `reset_processing_queues!`."""
function reset_iteration_queues!(state::VolPathState)
    concurrent_dispatch_group() do
        empty!(next_ray_queue(state))
        empty!(state.medium_sample_queue)
        empty!(state.medium_scatter_queue)
        empty!(state.hit_surface_queue)
        empty!(state.material_queue)
        empty!(state.shadow_queue)
        empty!(state.escaped_queue)
        empty!(state.hit_area_light_queue)
        state.per_material_queue !== nothing && empty!(state.per_material_queue)
    end
end

"""Reset film buffer for a new sample."""
function reset_film!(state::VolPathState)
    KA.fill!(state.pixel_L, 0f0)
end
