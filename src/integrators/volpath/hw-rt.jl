# Hikari Hardware RT Integration
#
# VolPath-specific dispatch overrides for hardware-accelerated ray tracing.
# Uses only Raycore types and stubs -- no backend (Lava/Metal) imports.
# Backend extensions in Raycore implement the actual dispatch.

import Raycore: HWTLAS, HWAdaptedAccel, RTRay, RTHitResult,
                trace_closest_hits!, trace_closest_hits_indirect!,
                batch_trace_indirect, set_custom_anyhit!,
                rt_primitive_id, rt_instance_custom_index, rt_instance_id, rt_launch_id_x,
                rt_ignore_intersection, rt_payload_store!, rt_payload_load, rt_trace_ray!

# Any backend with hw_accel=true creates an HWTLAS
default_accel(backend, ::Val{true}) = HWTLAS(backend)

# Scene sync! for HWTLAS scenes
function sync!(scene::Scene{<:HWTLAS})
    Raycore.sync!(scene.accel)
    bound = Raycore.world_bound(scene.accel)
    scene.bounds[] = (bound, bounding_sphere(bound))
    return scene
end

# ============================================================================
# fill_aux_buffers! for HWTLAS: trace primary rays via HW RT for depth
# ============================================================================

@kernel inbounds = true function hw_generate_primary_rays_kernel!(
    rays, @Const(camera), @Const(crop_p_min),
    @Const(width::Int32), @Const(height::Int32),
)
    idx = @index(Global)
    if idx <= width * height
        h = height
        row = Int32(((idx - Int32(1)) % h) + Int32(1))
        col = Int32(((idx - Int32(1)) ÷ h) + Int32(1))
        px = Float32(col) + crop_p_min[1] - 1f0
        py = Float32(row) + crop_p_min[2] - 1f0
        pixel = Point2f(px + 0.5f0, py + 0.5f0)
        cs = CameraSample(pixel, Point2f(0.5f0, 0.5f0), 0f0)
        ray, w = generate_ray(camera, cs)
        if w > 0f0
            o = ray.o; d = ray.d
            rays[idx] = RTRay(o[1], o[2], o[3], 0f0, d[1], d[2], d[3], 1f10)
        else
            rays[idx] = RTRay(0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 1f0, 0f0)
        end
    end
end

@kernel inbounds = true function hw_extract_depth_kernel!(
    depth, normal, albedo,
    @Const(results), @Const(miss_depth::Float32), @Const(n::Int32),
)
    idx = @index(Global)
    if idx <= n
        r = results[idx]
        if r.hit == UInt32(1)
            depth[idx] = r.t
            normal[idx] = Vec3f(0f0, 0f0, 1f0)
            albedo[idx] = RGB{Float32}(0.8f0, 0.8f0, 0.8f0)
        else
            depth[idx] = miss_depth
            normal[idx] = Vec3f(0f0, 0f0, 0f0)
            albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
        end
    end
end

function fill_aux_buffers!(film::Film, scene::Scene{<:HWAdaptedAccel}, camera; has_infinite_lights::Bool=false)
    hwtlas = scene.accel.hwtlas
    hwtlas.hw_accel === nothing && return nothing

    backend = KA.get_backend(film.depth)
    h, w = size(film.depth)
    n = h * w
    miss_depth = has_infinite_lights ? Float32(1e30) : Inf32

    # Cache rays/results buffers on the film. The framebuffer dimensions don't
    # change between resizes (resize creates a new Film), so length(film.depth)
    # is stable for the film's lifetime — one allocation per Film instance.
    rays = film.aux_rays[]
    if rays === nothing || length(rays::AbstractVector{RTRay}) != n
        rays = KA.allocate(backend, RTRay, n)
        film.aux_rays[] = rays
    end
    results = film.aux_results[]
    if results === nothing || length(results::AbstractVector{RTHitResult}) != n
        results = KA.allocate(backend, RTHitResult, n)
        film.aux_results[] = results
    end

    hw_generate_primary_rays_kernel!(backend)(
        rays, camera, film.crop_bounds.p_min, Int32(w), Int32(h); ndrange=n)
    trace_closest_hits!(results, rays, scene.accel, n)
    hw_extract_depth_kernel!(backend)(
        film.depth, film.normal, film.albedo, results, miss_depth, Int32(n); ndrange=n)
    KA.synchronize(backend)
    return film
end

# ============================================================================
# Ray Extraction Kernel
# ============================================================================

@kernel function extract_rays_kernel!(ray_buf, @Const(queue_items), @Const(queue_size))
    i = @index(Global)
    if i <= queue_size[1]
        work = queue_items[i]
        ray = work.ray
        ray_buf[i] = RTRay(
            Float32(ray.o[1]), Float32(ray.o[2]), Float32(ray.o[3]), 0.001f0,
            Float32(ray.d[1]), Float32(ray.d[2]), Float32(ray.d[3]), ray.t_max)
    end
end

# ============================================================================
# VolPath Dispatch Overrides
# ============================================================================

function detect_initial_medium(backend, accel::HWAdaptedAccel, mi, pos, vp::VolPath)
    return Raycore.SetKey()
end

function _get_hw_buf!(state::VolPathState, field::Symbol, T, cap)
    buf = getfield(state, field)
    if buf === nothing || length(buf) < cap
        buf = KA.allocate(state.backend, T, cap)
        setfield!(state, field, buf)
    end
    return buf
end

function vp_trace_rays!(state::VolPathState, accel::HWAdaptedAccel, media_interfaces, materials, ::VolPath)
    input_queue = state.current_ray_queue == :a ? state.ray_queue_a : state.ray_queue_b
    backend = KA.get_backend(input_queue.items)
    cap = Int(input_queue.capacity)

    ray_buf = _get_hw_buf!(state, :hw_primary_ray_buf, RTRay, cap)
    result_buf = _get_hw_buf!(state, :hw_primary_result_buf, RTHitResult, cap)

    extract_rays_kernel!(backend, 256)(ray_buf, input_queue.items, input_queue.size; ndrange=input_queue.size)

    precomputed = batch_trace_indirect(result_buf, ray_buf, accel, input_queue.size)

    foreach(vp_trace_rays_kernel!,
        input_queue,
        state.medium_sample_queue,
        state.escaped_queue,
        state.hit_surface_queue,
        precomputed,
        media_interfaces,
        materials,
    )
    return nothing
end

# ============================================================================
# Hardware-Accelerated Shadow Ray Tracing (Multi-Round Dispatch)
# ============================================================================

const _SS4 = SampledSpectrum{4}
const _SW4 = SampledWavelengths{4}
const _SK = Raycore.SetKey

struct ShadowIterState
    Ld::_SS4
    r_u_path::_SS4
    r_l_path::_SS4
    lambda::_SW4
    pixel_index::Int32
    ray_o::Point3f
    dir::Vec3f
    t_remaining::Float32
    medium_idx::_SK
    T_ray::_SS4
    tr_r_u::_SS4
    tr_r_l::_SS4
    active::UInt32
    visible::UInt32
end

@kernel inbounds=true function init_shadow_states_kernel!(
    states, @Const(queue_items), @Const(queue_size), @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        work = queue_items[i]
        states[i] = ShadowIterState(
            work.Ld, work.r_u, work.r_l, work.lambda, work.pixel_index,
            Point3f(work.ray.o), Vec3f(work.ray.d), work.t_max, work.medium_idx,
            _SS4(1f0), _SS4(1f0), _SS4(1f0), UInt32(1), UInt32(1))
    end
end

@kernel inbounds=true function extract_shadow_rays2_kernel!(
    ray_buf, @Const(states), @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        st = states[i]
        if st.active == UInt32(1) && st.t_remaining >= 1f-6
            ray_buf[i] = RTRay(
                Float32(st.ray_o[1]), Float32(st.ray_o[2]), Float32(st.ray_o[3]), 0.001f0,
                Float32(st.dir[1]), Float32(st.dir[2]), Float32(st.dir[3]), st.t_remaining)
        else
            ray_buf[i] = RTRay(0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 1f0, -1f0)
        end
    end
end

@kernel inbounds=true function process_shadow_round_kernel!(
    states, @Const(result_buf), @Const(tri_gpu), @Const(off_gpu),
    media_interfaces, media, materials, rgb2spec_table, @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        st = states[i]
        if st.active == UInt32(1)
            if st.t_remaining < 1f-6
                states[i] = ShadowIterState(
                    st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                    st.ray_o, st.dir, st.t_remaining, st.medium_idx,
                    st.T_ray, st.tr_r_u, st.tr_r_l, UInt32(0), UInt32(1))
            else
                result = result_buf[i]
                if result.hit == UInt32(0)
                    T_new, r_u_new, r_l_new = st.T_ray, st.tr_r_u, st.tr_r_l
                    if has_medium(st.medium_idx)
                        seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                            rgb2spec_table, media, st.medium_idx,
                            st.ray_o, st.dir, st.t_remaining, st.lambda)
                        T_new = T_new * seg_T; r_u_new = r_u_new * seg_r_u; r_l_new = r_l_new * seg_r_l
                    end
                    states[i] = ShadowIterState(
                        st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                        st.ray_o, st.dir, st.t_remaining, st.medium_idx,
                        T_new, r_u_new, r_l_new, UInt32(0), UInt32(1))
                else
                    # off_gpu is per-instance (keyed by gl_InstanceID = result.instance_id).
                    tri_idx = Int(off_gpu[result.instance_id + UInt32(1)]) + Int(result.primitive_id) + 1
                    tri = tri_gpu[tri_idx]
                    t_hit = result.t
                    w_bary = 1f0 - result.bary_u - result.bary_v
                    bary = SVector{3,Float32}(w_bary, result.bary_u, result.bary_v)

                    # instance_custom_index = interface override (0 = inherit from triangle).
                    mi_idx = result.instance_custom_index != UInt32(0) ?
                             result.instance_custom_index :
                             tri.metadata.medium_interface_idx
                    mi = media_interfaces[mi_idx]
                    ng = vp_compute_geometric_normal(tri)
                    entering = dot(Vec3f(st.dir), ng) < 0f0
                    is_xmit = is_medium_transition(mi)
                    handled = false

                    if is_xmit
                        T_new, r_u_new, r_l_new = st.T_ray, st.tr_r_u, st.tr_r_l
                        if has_medium(st.medium_idx)
                            seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                                rgb2spec_table, media, st.medium_idx,
                                st.ray_o, st.dir, t_hit, st.lambda)
                            T_new = T_new * seg_T; r_u_new = r_u_new * seg_r_u; r_l_new = r_l_new * seg_r_l
                        end
                        if is_black(T_new)
                            states[i] = ShadowIterState(
                                st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                                st.ray_o, st.dir, st.t_remaining, st.medium_idx,
                                T_new, r_u_new, r_l_new, UInt32(0), UInt32(1))
                        else
                            new_medium = get_crossing_medium(mi, entering)
                            new_ray_o = Point3f(st.ray_o + st.dir * (t_hit + 1f-4))
                            new_t_rem = st.t_remaining - t_hit - 1f-4
                            states[i] = ShadowIterState(
                                st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                                new_ray_o, st.dir, new_t_rem, new_medium,
                                T_new, r_u_new, r_l_new, UInt32(1), UInt32(1))
                        end
                        handled = true
                    end

                    if !is_xmit && !handled
                        uv = vp_compute_uv_barycentric(tri, bary)
                        mat_idx = mi.material
                        alpha = get_surface_alpha_dispatch(materials, mat_idx, uv)
                        alpha_pass = false

                        if alpha < 1f0
                            rng = pcg32_init(pbrt_hash(st.ray_o), pbrt_hash(st.dir))
                            alpha_u, _ = pcg32_uniform_f32(rng)
                            if alpha_u > alpha
                                T_new, r_u_new, r_l_new = st.T_ray, st.tr_r_u, st.tr_r_l
                                if has_medium(st.medium_idx)
                                    seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                                        rgb2spec_table, media, st.medium_idx,
                                        st.ray_o, st.dir, t_hit, st.lambda)
                                    T_new = T_new * seg_T; r_u_new = r_u_new * seg_r_u; r_l_new = r_l_new * seg_r_l
                                end
                                new_ray_o = Point3f(st.ray_o + st.dir * (t_hit + 1f-4))
                                new_t_rem = st.t_remaining - t_hit - 1f-4
                                states[i] = ShadowIterState(
                                    st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                                    new_ray_o, st.dir, new_t_rem, st.medium_idx,
                                    T_new, r_u_new, r_l_new, UInt32(1), UInt32(1))
                                alpha_pass = true
                            end
                        end

                        if !alpha_pass
                            states[i] = ShadowIterState(
                                st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                                st.ray_o, st.dir, st.t_remaining, st.medium_idx,
                                _SS4(0f0), _SS4(1f0), _SS4(1f0), UInt32(0), UInt32(0))
                        end
                    end
                end
            end
        end
    end
end

@kernel inbounds=true function count_active_shadows_kernel!(
    counter, @Const(states), @Const(n::Int32)
)
    i = @index(Global)
    if i <= n && states[i].active == UInt32(1)
        Atomix.@atomic counter[1] += Int32(1)
    end
end

@kernel inbounds=true function finalize_shadow_kernel!(
    states, pixel_L, @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        st = states[i]
        if st.visible == UInt32(1) && !is_black(st.T_ray)
            mis_weight = st.r_u_path * st.tr_r_u + st.r_l_path * st.tr_r_l
            mis_denom = average(mis_weight)
            if mis_denom > 1f-10
                final_L = st.Ld * st.T_ray / mis_denom
                if !is_black(final_L)
                    base_idx = (st.pixel_index - Int32(1)) * Int32(4)
                    accumulate_spectrum!(pixel_L, base_idx, final_L)
                end
            end
        end
    end
end

function vp_trace_shadow_rays!(state::VolPathState, accel::HWAdaptedAccel, media_interfaces, media, materials, vp::VolPath)
    hwtlas = accel.hwtlas
    shadow_queue = state.shadow_queue
    backend = KA.get_backend(shadow_queue.items)
    cap = Int(shadow_queue.capacity)

    shadow_states = _get_hw_buf!(state, :hw_shadow_states, ShadowIterState, cap)
    ray_buf = _get_hw_buf!(state, :hw_shadow_ray_buf, RTRay, cap)
    result_buf = _get_hw_buf!(state, :hw_shadow_result_buf, RTHitResult, cap)
    active_counter = _get_hw_buf!(state, :hw_shadow_counter, Int32, 1)
    n_rays_gpu = shadow_queue.size

    init_k! = init_shadow_states_kernel!(backend, 256)
    init_k!(shadow_states, shadow_queue.items, shadow_queue.size, Int32(cap); ndrange=n_rays_gpu)

    extract_k! = extract_shadow_rays2_kernel!(backend, 256)
    process_k! = process_shadow_round_kernel!(backend, 256)
    count_k! = count_active_shadows_kernel!(backend, 256)

    # Round 1
    extract_k!(ray_buf, shadow_states, Int32(cap); ndrange=n_rays_gpu)
    trace_closest_hits_indirect!(result_buf, ray_buf, accel, n_rays_gpu)
    process_k!(shadow_states, result_buf, hwtlas.tri_gpu, hwtlas.off_gpu,
               media_interfaces, media, materials, state.rgb2spec_table,
               Int32(cap); ndrange=n_rays_gpu)

    # Extra rounds for medium traversal
    max_shadow_rounds = isempty(media) ? 1 : 3
    for _round in 2:max_shadow_rounds
        fill!(active_counter, Int32(0))
        count_k!(active_counter, shadow_states, Int32(cap); ndrange=n_rays_gpu)
        extract_k!(ray_buf, shadow_states, Int32(cap); ndrange=n_rays_gpu)
        trace_closest_hits_indirect!(result_buf, ray_buf, accel, active_counter)
        process_k!(shadow_states, result_buf, hwtlas.tri_gpu, hwtlas.off_gpu,
                   media_interfaces, media, materials, state.rgb2spec_table,
                   Int32(cap); ndrange=n_rays_gpu)
    end

    finalize_shadow_kernel!(backend, 256)(shadow_states, state.pixel_L, Int32(cap); ndrange=n_rays_gpu)
    return nothing
end

# ============================================================================
# Any-Hit Shadow Shader (uses Raycore RT intrinsics, no backend imports)
# ============================================================================

function hw_raygen_shadow(accel, rays, results, tri_gpu, off_gpu, media_interfaces)
    lid = rt_launch_id_x(accel)
    ray = rays[lid + 1]

    rt_payload_store!(accel, 0f0, UInt32(0))
    rt_payload_store!(accel, -1f0, UInt32(1))

    rt_trace_ray!(accel,
        UInt32(0), UInt32(0xFF), UInt32(0), UInt32(0), UInt32(0),
        ray.origin_x, ray.origin_y, ray.origin_z, ray.tmin,
        ray.dir_x, ray.dir_y, ray.dir_z, ray.tmax)

    hit = rt_payload_load(accel, UInt32(0))
    t   = rt_payload_load(accel, UInt32(1))
    pid = rt_payload_load(accel, UInt32(2))
    ci  = rt_payload_load(accel, UInt32(3))
    bu  = rt_payload_load(accel, UInt32(4))
    bv  = rt_payload_load(accel, UInt32(5))
    iid = rt_payload_load(accel, UInt32(6))

    results[lid + 1] = RTHitResult(
        reinterpret(UInt32, hit), t,
        reinterpret(UInt32, pid), reinterpret(UInt32, ci),
        bu, bv,
        reinterpret(UInt32, iid),   # gl_InstanceID (0-based)
        UInt32(0))
    return nothing
end

function hw_anyhit_shadow(accel, tri_gpu, off_gpu, media_interfaces)
    prim_id  = rt_primitive_id(accel)
    override = rt_instance_custom_index(accel)   # interface override
    iid      = rt_instance_id(accel)             # 0-based instance index → triangle lookup
    tri_offset = off_gpu[iid + UInt32(1)]
    tri = tri_gpu[Int(tri_offset) + Int(prim_id) + 1]
    mi_idx = override != UInt32(0) ? override : tri.metadata.medium_interface_idx
    mi = media_interfaces[Int(mi_idx)]
    if mi.inside.type_idx != mi.outside.type_idx || mi.inside.vec_idx != mi.outside.vec_idx
        rt_ignore_intersection(accel)
    end
    return nothing
end

function ensure_anyhit_pipeline!(accel::HWAdaptedAccel)
    set_custom_anyhit!(accel, hw_anyhit_shadow, hw_raygen_shadow)
end
