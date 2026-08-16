# Surface material evaluation for VolPath
# Handles BSDF sampling, direct lighting, and path continuation at surface hits
#
# NOTE: `materials` is a StaticMultiTypeSet containing both materials and textures.
# Use eval_tex(materials, field, uv) to sample textures via TextureRef.

# ============================================================================
# Texture Filtering Derivatives (pbrt-v4 style)
# ============================================================================

"""
    approximate_dp_dxy(pi, n, camera, samples_per_pixel) -> (dpdx, dpdy)

Approximate screen-space position derivatives at intersection point.
Following pbrt-v4's Camera::Approximate_dp_dxy method.

This estimates how much the surface position changes per pixel, which is used
for texture filtering (mipmap level selection). The approximation assumes the
surface is locally planar near the intersection point.

For a perspective camera, this is approximately:
    dp/dscreen ≈ distance * tan(fov/2) / (resolution/2)

Arguments:
- `pi`: Intersection point in world space
- `n`: Surface normal at intersection
- `camera`: Camera with dx_camera, dy_camera precomputed
- `samples_per_pixel`: Number of samples per pixel (for scaling)

Returns (dpdx, dpdy) - approximate change in position per screen pixel.
"""
@propagate_inbounds function approximate_dp_dxy(
    pi::Point3f, n::Vec3f, camera, samples_per_pixel::Int32
)
    # Get camera_to_world transform (generic accessor handles different camera types)
    camera_to_world = get_camera_to_world(camera)

    # For perspective camera, approximate dpdx/dpdy based on distance and fov
    # The camera has precomputed dx_camera and dy_camera (change per pixel in camera space)

    # Get camera position (world space origin)
    camera_pos = Raycore.transform_point(camera_to_world.m, Point3f(0f0))

    # Distance from camera to intersection point
    to_point = Vec3f(pi - camera_pos)
    dist = sqrt(dot(to_point, to_point))

    # Hikari's `dx_camera` is the camera-space per-raster-pixel displacement
    # on the perspective NEAR plane (z = PERSPECTIVE_NEAR), not the angular
    # size of a pixel. To project it to world displacement at the actual hit
    # distance we have to multiply by `dist / near`; otherwise the world
    # displacement comes out ~100× too small and the per-pixel UV footprint
    # falls below BUMP_DEFAULT_DELTA — so the bump fallback (5e-4) kicks in
    # instead of the real screen-space derivative, leaving the gold-dome
    # conductor with coherent mirror highlights.
    #
    # pbrt-v4 scales BOTH differential paths by max(.125, 1/sqrt(spp)):
    # true camera ray diffs via `ray.ScaleDifferentials(rayDiffScale)`
    # (cpu/integrators.cpp:251) and the fallback via `sppScale` inside
    # `Approximate_dp_dxy` (cameras.h:175). An earlier comment here claimed
    # the exact-diff path is unscaled and skipped sppScale — that left the
    # footprint 8× too large at 256 spp and over-smoothed every bump map
    # (shadow_bumpgold_dome_over_velvet pins this). The sub-texel-du
    # concern that motivated skipping it only applied to the old
    # nearest-neighbour `sample_texture_data`; bilinear sampling returns
    # the correct local slope for sub-texel finite differences.
    spp_scale = max(0.125f0, 1f0 / sqrt(Float32(samples_per_pixel)))
    scale = spp_scale * dist / PERSPECTIVE_NEAR

    # Transform dx_camera and dy_camera to world space
    # These represent how the ray direction changes per pixel
    dx_world = Raycore.transform_direction(camera_to_world.m, camera.dx_camera)
    dy_world = Raycore.transform_direction(camera_to_world.m, camera.dy_camera)

    # Project onto the tangent plane at the intersection
    # dpdx ≈ scale * (dx_world - n * dot(n, dx_world))
    dpdx = scale * (dx_world - n * dot(n, dx_world))
    dpdy = scale * (dy_world - n * dot(n, dy_world))

    return dpdx, dpdy
end

"""
    compute_uv_derivatives(dpdu, dpdv, dpdx, dpdy) -> (dudx, dudy, dvdx, dvdy)

Compute UV derivatives from position derivatives using least-squares solve.
Following pbrt-v4's SurfaceInteraction::ComputeDifferentials.

Given:
- dpdu, dpdv: How position changes with UV (∂p/∂u, ∂p/∂v)
- dpdx, dpdy: How position changes with screen pixel (∂p/∂x, ∂p/∂y)

Solve for:
- dudx, dudy: How u changes with screen pixel (∂u/∂x, ∂u/∂y)
- dvdx, dvdy: How v changes with screen pixel (∂v/∂x, ∂v/∂y)

Uses the normal equations: (A^T A) [du/dx; dv/dx]^T = A^T [dpdx]
where A = [dpdu | dpdv] is a 3x2 matrix.
"""
@propagate_inbounds function compute_uv_derivatives(
    dpdu::Vec3f, dpdv::Vec3f, dpdx::Vec3f, dpdy::Vec3f
)
    # Compute A^T A (2x2 matrix)
    ata00 = dot(dpdu, dpdu)
    ata01 = dot(dpdu, dpdv)
    ata11 = dot(dpdv, dpdv)

    # Compute determinant and check for degeneracy
    det = ata00 * ata11 - ata01 * ata01
    if abs(det) < 1f-10
        return (0f0, 0f0, 0f0, 0f0)
    end
    inv_det = 1f0 / det

    # Compute A^T b for x direction
    atb0x = dot(dpdu, dpdx)
    atb1x = dot(dpdv, dpdx)

    # Compute A^T b for y direction
    atb0y = dot(dpdu, dpdy)
    atb1y = dot(dpdv, dpdy)

    # Solve using Cramer's rule (2x2 system)
    # [dudx]   [ata11  -ata01] [atb0x]
    # [dvdx] = [-ata01  ata00] [atb1x] * inv_det
    dudx = (ata11 * atb0x - ata01 * atb1x) * inv_det
    dvdx = (ata00 * atb1x - ata01 * atb0x) * inv_det
    dudy = (ata11 * atb0y - ata01 * atb1y) * inv_det
    dvdy = (ata00 * atb1y - ata01 * atb0y) * inv_det

    # Clamp to reasonable values (following pbrt-v4)
    clamp_val = 1f8
    dudx = clamp(dudx, -clamp_val, clamp_val)
    dvdx = clamp(dvdx, -clamp_val, clamp_val)
    dudy = clamp(dudy, -clamp_val, clamp_val)
    dvdy = clamp(dvdy, -clamp_val, clamp_val)

    return (dudx, dudy, dvdx, dvdy)
end

# TextureFilterContext is defined in textures/texture-ref.jl

"""
    compute_texture_filter_context(work, camera, samples_per_pixel) -> TextureFilterContext

Compute texture filtering context from material evaluation work item.
Uses approximate screen-space derivatives for proper mipmap selection.
"""
@propagate_inbounds function compute_texture_filter_context(
    work::VPMaterialEvalWorkItem, camera, samples_per_pixel::Int32
)
    # Compute screen-space position derivatives
    dpdx, dpdy = approximate_dp_dxy(work.pi, work.n, camera, samples_per_pixel)

    # Compute UV derivatives from position derivatives
    dudx, dudy, dvdx, dvdy = compute_uv_derivatives(work.dpdu, work.dpdv, dpdx, dpdy)

    return TextureFilterContext(work.uv, dudx, dudy, dvdx, dvdy, work.face_idx, work.bary)
end

# (vp_process_surface_hits! and its kernel were deleted: the fused
# trace-and-shade path made them dead, and with them the intermediate
# `material_queue` — 379 MiB of vestigial allocation on a 1.4 Mpx render.)

# ============================================================================
# Direct Lighting Inner Function
# ============================================================================

"""Inner function for surface direct lighting - can use return statements.

Uses BVH light sampler for spatially-aware importance sampling.
Nearby lights get higher probability than distant ones (pbrt-v4's BVHLightSampler).

Sobol samples are generated inline from `(px, py, sample_idx, base_dim+dim)`
rather than read from a pre-populated per-pixel buffer. Eliminates the
separate `vp_generate_ray_samples_kernel!` dispatch + barrier, plus the
SOA write/read of two `pixel_samples_direct_*` arrays per bounce — measured
as 58 % of GPU time on killeroo before this refactor.
"""
@propagate_inbounds function surface_direct_lighting_inner!(
    pixel_L,
    accel,
    media_interfaces,
    media,
    work::VPMaterialEvalWorkItem,
    materials,
    lights,
    rgb2spec_table,
    bvh_nodes,
    infinite_light_indices,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    num_lights::Int32,
    sobol_rng,           # SobolRNG — samples are generated on demand inline
    sample_idx::Int32,   # which sample of samples_per_pixel we're on
    # Camera for texture filtering (pbrt-v4 style)
    camera,
    samples_per_pixel::Int32,
    do_regularize::Bool
)
    # Skip if no lights
    num_lights < Int32(1) && return

    # Skip null-material boundaries (pbrt `Material "interface"` / nullptr):
    # no BSDF to evaluate, no direct lighting contribution at this surface.
    # `evaluate_material_inner!` advances the ray past the boundary.
    if !Raycore.is_valid(work.material_idx) && is_medium_transition(work.interface)
        return
    end

    # Inline Sobol sample generation.  Dimension allocation matches the
    # original `vp_generate_ray_samples_kernel!`: 6 (camera) + 7 * depth +
    # dim-offset.  Direct lighting uses dim+1 (1D light select) and dim+3
    # (2D light position).
    pixel_idx = work.pixel_index
    pixel_idx_0 = pixel_idx - Int32(1)
    px = u_mod(pixel_idx_0, sobol_rng.width) + Int32(1)
    py = u_div(pixel_idx_0, sobol_rng.width) + Int32(1)
    base_dim = Int32(6) + Int32(7) * work.depth
    light_select = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(1))
    u_light_x, u_light_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(3))
    u_light = Point2f(u_light_x, u_light_y)

    # Select light using BVH importance-weighted sampling
    # Returns (1-based flat index, PMF for that light)
    light_idx, light_pmf = bvh_sample_light(
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights,
        work.pi, work.ns, light_select
    )

    # Validate index (should always be valid if sampler was built correctly)
    if light_idx < Int32(1) || light_idx > num_lights || light_pmf <= 0f0
        return
    end

    # Sample the light (works with both Tuple and StaticMultiTypeSet)
    light_sample = sample_light_spectral(
        rgb2spec_table, lights, light_idx, work.pi, work.lambda, u_light
    )

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        # Compute texture filter context with proper screen-space derivatives (pbrt-v4 style)
        tfc = compute_texture_filter_context(work, camera, samples_per_pixel)

        # Apply regularization if enabled and we've had a non-specular bounce
        # Must match the state used during BSDF sampling (pbrt-v4: Regularize() modifies BxDF in-place)
        regularize = do_regularize && work.any_non_specular_bounces

        # Evaluate BSDF for light direction
        bsdf_f, bsdf_pdf = evaluate_spectral_material(
            rgb2spec_table, materials, work.material_idx,
            work.wo, light_sample.wi, work.ns, work.dpdus, tfc, work.lambda, regularize
        )

        if !is_black(bsdf_f)
            # Compute direct lighting contribution with MIS
            result = compute_direct_lighting_spectral(
                work.pi, work.n, work.ns, work.wo, work.beta, work.r_u, work.lambda,
                light_sample, bsdf_f, bsdf_pdf
            )

            if result.valid
                # Following pbrt-v4 (surfscatter.cpp lines 299-315):
                # - Ld = beta * f * Li (no PDF division - that happens at shadow ray resolution)
                # - r_l = r_u * lightPDF where lightPDF = ls.pdf * light_pmf
                # compute_direct_lighting_spectral already sets r_l = r_u * ls.pdf
                # So we multiply by light_pmf to get the full light PDF in r_l
                scaled_r_l = result.r_l * light_pmf

                # Determine medium for shadow ray based on direction at medium transitions
                # (mirrors pbrt-v4 SurfaceInteraction::GetMedium(w))
                shadow_medium = if is_medium_transition(work.interface)
                    get_medium_index(work.interface, result.ray_direction, work.n)
                else
                    work.current_medium
                end

                # ── Inline shadow trace + accumulate (was: push to shadow_queue, run
                # vp_trace_shadow_rays! as a separate dispatch). Body mirrors
                # vp_trace_shadow_rays_kernel! exactly: trace transmittance, combine
                # path MIS weights with transmittance MIS weights, accumulate to
                # pixel_L. Eliminates the queue write + read + per-bounce shadow
                # dispatch + barrier for every surface direct-lighting contribution.
                T_ray, tr_r_u, tr_r_l, visible = trace_shadow_transmittance(
                    accel, media_interfaces, media, materials, rgb2spec_table,
                    result.ray_origin, result.ray_direction, result.t_max,
                    work.lambda, shadow_medium,
                )

                if visible && !is_black(T_ray)
                    mis_weight = result.r_u * tr_r_u + scaled_r_l * tr_r_l
                    mis_denom = average(mis_weight)
                    if mis_denom > 1f-10
                        final_L = result.Ld * T_ray / mis_denom
                        if !is_black(final_L)
                            base_idx = (work.pixel_index - Int32(1)) * Int32(4)
                            accumulate_spectrum!(pixel_L, base_idx, final_L)
                        end
                    end
                end
            end
        end
    end
    return
end

# `vp_sample_surface_direct_lighting!` was deleted by commit cb3d08f's
# fusion (process_hits + direct_light + evaluate_materials → one kernel),
# and the inline-shadow-trace optimisation in this commit folded the shadow
# tracing into the same kernel too, so there's no longer any standalone
# direct-lighting dispatch on the surface path. Light sampling + BSDF eval
# for direct lighting + shadow trace + pixel_L accumulation all happen
# inside `vp_shade_surface_hits_kernel!`'s call to
# `surface_direct_lighting_inner!`.

# ============================================================================
# BSDF Sampling Inner Function
# ============================================================================

"""Inner function for material evaluation - can use return statements.

Sobol samples are generated inline from `(px, py, sample_idx, base_dim+dim)`
rather than read from a pre-populated per-pixel buffer.  See
`surface_direct_lighting_inner!` for the same change on the direct path.
"""
@propagate_inbounds function evaluate_material_inner!(
    next_ray_queue,
    work::VPMaterialEvalWorkItem,
    materials,
    rgb2spec_table,
    max_depth::Int32,
    do_regularize::Bool,
    sobol_rng,           # SobolRNG — samples generated on demand inline
    sample_idx::Int32,
    # Camera for texture filtering (pbrt-v4 style)
    camera,
    samples_per_pixel::Int32,
    rr_depth::Int32
)
    # ── Null-material skip (pbrt `Material "interface"` / nullptr) ──
    # No BSDF; rays pass through with medium swap only and consume no path depth.
    # Mirrors pbrt cpu/integrators.cpp:420,568,681 `if (!bsdf) SkipIntersection(...)`.
    if !Raycore.is_valid(work.material_idx) && is_medium_transition(work.interface)
        ray_d = -work.wo
        new_medium = get_medium_index(work.interface, ray_d, work.n)
        offset_dir = if dot(ray_d, work.n) > 0f0; work.n; else; -work.n; end
        ray_origin = Point3f(work.pi + offset_dir * 1f-4)
        new_ray = Raycore.Ray(o=ray_origin, d=ray_d, t_max=Inf32, time=0f0)
        ray_item = VPRayWorkItem(
            new_ray, work.depth,                    # depth NOT incremented
            work.lambda, work.pixel_index,
            work.beta, work.r_u, work.r_l,
            work.prev_intr_p, work.prev_intr_n,
            work.eta_scale,
            work.specular_bounce, work.any_non_specular_bounces,
            new_medium)
        push!(next_ray_queue, ray_item)
        return
    end

    # Check depth limit
    new_depth = work.depth + Int32(1)
    if new_depth >= max_depth
        return
    end

    # Inline Sobol sample generation.  Indirect uses dim+4 (BSDF component
    # select), dim+6 (2D direction), dim+7 (RR).
    pixel_idx = work.pixel_index
    pixel_idx_0 = pixel_idx - Int32(1)
    px = u_mod(pixel_idx_0, sobol_rng.width) + Int32(1)
    py = u_div(pixel_idx_0, sobol_rng.width) + Int32(1)
    base_dim = Int32(6) + Int32(7) * work.depth
    rng = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(4))
    u_x, u_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(6))
    u = Point2f(u_x, u_y)
    rr_sample = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(7))

    # Apply regularization if enabled and we've had a non-specular bounce
    # (pbrt-v4: regularize && anyNonSpecularBounces)
    regularize = do_regularize && work.any_non_specular_bounces

    # Compute texture filter context with proper screen-space derivatives (pbrt-v4 style)
    tfc = compute_texture_filter_context(work, camera, samples_per_pixel)

    # Sample BSDF
    sample = sample_spectral_material(
        rgb2spec_table, materials, work.material_idx,
        work.wo, work.ns, work.dpdus, tfc, work.lambda, u, rng, regularize
    )

    # Check if valid sample
    if sample.pdf > 0f0 && !is_black(sample.f)
        # pbrt surfscatter.cpp:190-203
        # beta always uses sample.pdf (proportional or exact)
        # r_l uses re-evaluated PDF when pdfIsProportional
        cos_theta = abs(dot(sample.wi, work.ns))
        new_beta = work.beta * sample.f * cos_theta / sample.pdf

        # Update eta scale for refraction — pbrt-v4 surfscatter.cpp:206-208
        # etaScale *= Sqr(bsdfSample->eta) only for transmission
        new_eta_scale = if is_transmissive(sample.flags)
            work.eta_scale * sample.eta * sample.eta
        else
            work.eta_scale
        end

        # Update MIS weights — pbrt surfscatter.cpp:200-203
        # pdfIsProportional: use re-evaluated PDF for MIS weight
        # otherwise: use sample.pdf
        r_l_pdf = if sample.pdf_is_proportional
            _, p = evaluate_spectral_material(
                rgb2spec_table, materials, work.material_idx,
                work.wo, sample.wi, work.ns, work.dpdus, tfc, work.lambda, regularize)
            max(p, 1f-10)
        else
            sample.pdf
        end
        new_r_l = work.r_u / r_l_pdf

        # Russian roulette
        should_continue, final_beta = russian_roulette_spectral(
            new_beta, work.r_u, new_eta_scale, new_depth, rr_sample, rr_depth
        )

        if should_continue
            # Determine medium for continuation ray using MediumInterfaceIdx
            # Following pbrt-v4: use ray direction relative to surface normal
            # to determine which medium the ray enters
            new_medium = if is_medium_transition(work.interface)
                # Surface defines a medium boundary - get medium based on ray direction
                # If wi · n > 0, ray goes "outside" the surface
                # If wi · n < 0, ray goes "inside" the surface
                get_medium_index(work.interface, sample.wi, work.n)
            else
                # No medium transition at this surface
                # Stay in current medium (reflection or transmission through regular material)
                work.current_medium
            end

            # Create continuation ray
            # Offset origin slightly to avoid self-intersection
            offset_dir = if dot(sample.wi, work.n) > 0f0
                work.n
            else
                -work.n
            end
            ray_origin = Point3f(work.pi + offset_dir * 1f-4)

            new_ray = Raycore.Ray(
                o = ray_origin,
                d = sample.wi,
                t_max = Inf32,
                time = 0f0
            )

            # Terminate secondary wavelengths for dispersive refraction (pbrt-v4)
            new_lambda = if sample.secondary_terminated
                terminate_secondary_wavelengths(work.lambda)
            else
                work.lambda
            end

            ray_item = VPRayWorkItem(
                new_ray,
                new_depth,
                new_lambda,
                work.pixel_index,
                final_beta,
                work.r_u,  # r_u unchanged
                new_r_l,
                work.pi,   # prev_intr_p
                work.ns,   # prev_intr_n
                new_eta_scale,
                is_specular(sample.flags),
                work.any_non_specular_bounces || !is_specular(sample.flags),
                new_medium
            )

            push!(next_ray_queue, ray_item)
        end
    end
    return
end

# (vp_evaluate_materials! and its kernel were deleted along with
# `material_queue` — the fused trace-and-shade path made them dead.)

# ============================================================================
# Fused Surface Shading Kernel
# ============================================================================
#
# Fuses `vp_process_surface_hits` + `vp_sample_surface_direct_lighting` +
# `vp_evaluate_materials` into a single dispatch. Each thread reads one
# `VPHitSurfaceWorkItem`, computes wo + material_idx, accumulates emission,
# then drives both `surface_direct_lighting_inner!` (pushes shadow ray) and
# `evaluate_material_inner!` (pushes continuation ray) with a stack-local
# `VPMaterialEvalWorkItem`. The intermediate `material_queue` is bypassed,
# eliminating ~170 MB of memory traffic per bounce on a 1.4M-pixel scene.

@propagate_inbounds function vp_shade_surface_hits_kernel!(
    work,
    next_ray_queue,
    pixel_L,
    accel,
    media_interfaces,
    media,
    materials,
    lights,
    rgb2spec_table,
    bvh_nodes,
    infinite_light_indices,
    light_to_bit_trail,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    num_lights::Int32,
    max_depth::Int32,
    do_regularize::Bool,
    sobol_rng, sample_idx::Int32,
    camera,
    samples_per_pixel::Int32,
    rr_depth::Int32,
)
    wo = -work.ray.d

    material_idx = resolve_mix_material(
        materials, work.material_idx,
        work.pi, wo, work.uv
    )

    # HandleEmissiveIntersection moved to `vp_handle_emitters_kernel!`
    # (matches pbrt-v4 hitAreaLightQueue). This kernel is now dead in the
    # live volpath loop — the fused trace-and-shade path bypasses it.

    # Synthesize stack-local material eval work item. This is the same shape
    # that used to be pushed into `material_queue` and re-read by both kernels.
    mat_work = VPMaterialEvalWorkItem(work, wo, material_idx)

    # ── Direct lighting + inline shadow trace + accumulate
    surface_direct_lighting_inner!(
        pixel_L, accel, media_interfaces, media,
        mat_work, materials, lights, rgb2spec_table,
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights, num_lights,
        sobol_rng, sample_idx,
        camera, samples_per_pixel,
        do_regularize,
    )

    # ── BSDF sample + RR + push continuation (was vp_evaluate_materials) ──
    evaluate_material_inner!(
        next_ray_queue,
        mat_work, materials, rgb2spec_table, max_depth,
        do_regularize,
        sobol_rng, sample_idx,
        camera, samples_per_pixel,
        rr_depth,
    )
    return
end

function vp_shade_surface_hits!(state::VolPathState, accel, media_interfaces, media,
                                materials, lights,
                                sample_idx::Int32,
                                camera, samples_per_pixel::Int32,
                                regularize::Bool = true)
    foreach(vp_shade_surface_hits_kernel!,
        state.hit_surface_queue,
        next_ray_queue(state),
        state.pixel_L,
        accel,
        media_interfaces,
        media,
        materials,
        lights,
        state.rgb2spec_table,
        state.bvh_nodes,
        state.infinite_light_indices,
        state.light_to_bit_trail,
        state.num_infinite_lights,
        state.num_bvh_lights,
        state.num_lights,
        state.max_depth,
        regularize,
        state.sobol_rng, sample_idx,
        camera, samples_per_pixel,
        state.rr_depth,
    )
    return nothing
end


# ============================================================================
# Per-material-type Shading (pbrt-v4 wavefront/surfscatter.cpp:41 pattern)
# ============================================================================
#
# `vp_shade_surface_hits_kernel!` above includes every concrete material's
# BSDF code (via `with_index(materials, ...)`) inside a single monolithic
# SPIR-V module.  Driver-reported register count was 128 on RTX 4000 Ada —
# right at the occupancy cliff (above 128 the per-SM thread count drops
# from 512 to 256).  pbrt-v4 sidesteps this by splitting shading into one
# kernel per concrete material type via `ForEachType`; each kernel is
# monomorphised on a single `ConcreteMaterial`, so its SPIR-V only contains
# that one material's BSDF.
#
# Pieces:
#   * `surface_direct_lighting_inner_typed!` / `evaluate_material_inner_typed!`
#     — same logic as the `_inner!` variants above but take the concrete
#     material `mat::M` and call `evaluate_bsdf_spectral(mat, ...)` /
#     `sample_bsdf_spectral(mat, ...)` directly.  No `with_index` for the
#     material axis.  Sobol inlined + shadow trace inlined, just like the
#     non-typed versions.
#   * `vp_shade_material_kernel!` — reads `TypedHitRef{T}`, loads the hit from
#     the shared `hit_surface_queue`, looks up the
#     concrete instance via `material_of_type(materials, T, vec_idx)` (one
#     array load resolved at compile time), accumulates emission, runs DL
#     + indirect path.
#   * `vp_shade_typed!` — `foreach_type` over the `MultiTypeMaterialQueue`,
#     wrapped in `Lava.concurrent_dispatch_group` so per-type kernels can
#     overlap on idle SMs (otherwise serialized by the per-dispatch
#     barriers — each per-type kernel writes to disjoint slots of the
#     shared `next_ray_queue` + atomic `pixel_L`).

"""Typed counterpart to `surface_direct_lighting_inner!`.  Same body but
calls `evaluate_bsdf_spectral(mat, ...)` directly instead of going through
`with_index`.  Caller passes the concrete material instance picked up via
`material_of_type` in the per-type kernel."""
@propagate_inbounds function surface_direct_lighting_inner_typed!(
    mat::M,
    pixel_L,
    accel,
    media_interfaces,
    media,
    work::VPMaterialEvalWorkItem,
    materials,
    lights,
    rgb2spec_table,
    bvh_nodes,
    infinite_light_indices,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    num_lights::Int32,
    sobol_rng,
    sample_idx::Int32,
    camera,
    samples_per_pixel::Int32,
    do_regularize::Bool,
) where M
    num_lights < Int32(1) && return

    # Null-material boundaries are routed away from the typed queues at the
    # push site (`vp_trace_kernel!` handles the medium swap inline).  A
    # concrete `mat::M` here is always a real BSDF.

    # Inline Sobol (matches `surface_direct_lighting_inner!` dim allocation).
    pixel_idx = work.pixel_index
    pixel_idx_0 = pixel_idx - Int32(1)
    px = u_mod(pixel_idx_0, sobol_rng.width) + Int32(1)
    py = u_div(pixel_idx_0, sobol_rng.width) + Int32(1)
    base_dim = Int32(6) + Int32(7) * work.depth
    light_select = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(1))
    u_light_x, u_light_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(3))
    u_light = Point2f(u_light_x, u_light_y)

    light_idx, light_pmf = bvh_sample_light(
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights,
        work.pi, work.ns, light_select,
    )
    if light_idx < Int32(1) || light_idx > num_lights || light_pmf <= 0f0
        return
    end

    light_sample = sample_light_spectral(
        rgb2spec_table, lights, light_idx, work.pi, work.lambda, u_light,
    )

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        tfc = compute_texture_filter_context(work, camera, samples_per_pixel)
        regularize = do_regularize && work.any_non_specular_bounces

        # Direct call into the concrete material — Julia inlines.
        bsdf_f, bsdf_pdf = evaluate_bsdf_spectral(
            mat, rgb2spec_table, materials,
            work.wo, light_sample.wi, work.ns, work.dpdus, tfc, work.lambda, regularize,
        )

        if !is_black(bsdf_f)
            result = compute_direct_lighting_spectral(
                work.pi, work.n, work.ns, work.wo, work.beta, work.r_u, work.lambda,
                light_sample, bsdf_f, bsdf_pdf,
            )
            if result.valid
                scaled_r_l = result.r_l * light_pmf

                shadow_medium = if is_medium_transition(work.interface)
                    get_medium_index(work.interface, result.ray_direction, work.n)
                else
                    work.current_medium
                end

                # Inline shadow trace + accumulate (opt4 pattern).
                T_ray, tr_r_u, tr_r_l, visible = trace_shadow_transmittance(
                    accel, media_interfaces, media, materials, rgb2spec_table,
                    result.ray_origin, result.ray_direction, result.t_max,
                    work.lambda, shadow_medium,
                )

                if visible && !is_black(T_ray)
                    mis_weight = result.r_u * tr_r_u + scaled_r_l * tr_r_l
                    mis_denom = average(mis_weight)
                    if mis_denom > 1f-10
                        final_L = result.Ld * T_ray / mis_denom
                        if !is_black(final_L)
                            base_idx = (work.pixel_index - Int32(1)) * Int32(4)
                            accumulate_spectrum!(pixel_L, base_idx, final_L)
                        end
                    end
                end
            end
        end
    end
    return
end

"""Typed counterpart to `evaluate_material_inner!`.  Uses
`sample_bsdf_spectral(mat, ...)` and `evaluate_bsdf_spectral(mat, ...)`
directly; no `with_index` for materials."""
@propagate_inbounds function evaluate_material_inner_typed!(
    mat::M,
    next_ray_queue,
    work::VPMaterialEvalWorkItem,
    materials,
    rgb2spec_table,
    max_depth::Int32,
    do_regularize::Bool,
    sobol_rng,
    sample_idx::Int32,
    camera,
    samples_per_pixel::Int32,
    rr_depth::Int32,
) where M
    # Null material → never sees a typed kernel (routed at push site).
    new_depth = work.depth + Int32(1)
    if new_depth >= max_depth
        return
    end

    pixel_idx = work.pixel_index
    pixel_idx_0 = pixel_idx - Int32(1)
    px = u_mod(pixel_idx_0, sobol_rng.width) + Int32(1)
    py = u_div(pixel_idx_0, sobol_rng.width) + Int32(1)
    base_dim = Int32(6) + Int32(7) * work.depth
    rng = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(4))
    u_x, u_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(6))
    u = Point2f(u_x, u_y)
    rr_sample = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(7))

    regularize = do_regularize && work.any_non_specular_bounces
    tfc = compute_texture_filter_context(work, camera, samples_per_pixel)

    sample = sample_bsdf_spectral(
        mat, rgb2spec_table, materials,
        work.wo, work.ns, work.dpdus, tfc, work.lambda, u, rng, regularize,
    )

    if sample.pdf > 0f0 && !is_black(sample.f)
        cos_theta = abs(dot(sample.wi, work.ns))
        new_beta = work.beta * sample.f * cos_theta / sample.pdf

        new_eta_scale = if is_transmissive(sample.flags)
            work.eta_scale * sample.eta * sample.eta
        else
            work.eta_scale
        end

        r_l_pdf = if sample.pdf_is_proportional
            _, p = evaluate_bsdf_spectral(
                mat, rgb2spec_table, materials,
                work.wo, sample.wi, work.ns, work.dpdus, tfc, work.lambda, regularize,
            )
            max(p, 1f-10)
        else
            sample.pdf
        end
        new_r_l = work.r_u / r_l_pdf

        should_continue, final_beta = russian_roulette_spectral(
            new_beta, work.r_u, new_eta_scale, new_depth, rr_sample, rr_depth,
        )

        if should_continue
            new_medium = if is_medium_transition(work.interface)
                get_medium_index(work.interface, sample.wi, work.n)
            else
                work.current_medium
            end
            offset_dir = if dot(sample.wi, work.n) > 0f0; work.n; else; -work.n; end
            ray_origin = Point3f(work.pi + offset_dir * 1f-4)
            new_ray = Raycore.Ray(o=ray_origin, d=sample.wi, t_max=Inf32, time=0f0)
            new_lambda = if sample.secondary_terminated
                terminate_secondary_wavelengths(work.lambda)
            else
                work.lambda
            end
            ray_item = VPRayWorkItem(
                new_ray, new_depth, new_lambda, work.pixel_index,
                final_beta, work.r_u, new_r_l,
                work.pi, work.ns,
                new_eta_scale,
                is_specular(sample.flags),
                work.any_non_specular_bounces || !is_specular(sample.flags),
                new_medium,
            )
            push!(next_ray_queue, ray_item)
        end
    end
    return
end

"""Per-material-type shading kernel — drains one slot of the
`MultiTypeMaterialQueue`.  Julia specialises this function separately for
each concrete `T`, so each per-type dispatch's SPIR-V contains exactly one
material's BSDF code — no `with_index` switching on the material axis.

Emission MIS still uses `with_index` on `lights` (light-type axis, separate
concern from materials)."""
@propagate_inbounds function vp_shade_material_kernel!(
    typed::TypedHitRef{T},
    hit_surface_queue,
    next_ray_queue,
    pixel_L,
    accel,
    media_interfaces,
    media,
    materials,
    lights,
    rgb2spec_table,
    bvh_nodes,
    infinite_light_indices,
    light_to_bit_trail,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    num_lights::Int32,
    max_depth::Int32,
    do_regularize::Bool,
    sobol_rng,
    sample_idx::Int32,
    camera,
    samples_per_pixel::Int32,
    rr_depth::Int32,
) where T
    # The typed queue carries 4-byte indices; the hit payload lives once in
    # the shared hit_surface_queue.
    work = hit_surface_queue.items[typed.idx]
    wo = -work.ray.d

    # Concrete material instance — compile-time slot lookup + indexed array
    # load.  No `with_index` switch in this kernel's SPIR-V.
    mat = material_of_type(materials, T, work.material_idx.vec_idx)

    # Emission MIS for indirect rays hitting area lights now runs in its own
    # kernel (`vp_handle_emitters_kernel!` draining `hit_area_light_queue`),
    # matching pbrt-v4 wavefront. The per-material kernels stay focused on
    # BSDF sampling + direct lighting.

    # Stack-local material-eval work item.
    mat_work = VPMaterialEvalWorkItem(work, wo, work.material_idx)

    # pbrt-v4 wavefront `EvaluateMaterialAndBSDF<ConcreteMaterial>` builds the
    # BxDF once per surface hit via `Material::GetBxDF` so the subsequent
    # `Sample_f` and `f` calls operate on a small struct with already-resolved
    # spectra / textures.  Default `get_bxdf` is identity, so materials whose
    # BSDF state is already small (Diffuse: 16 B) pay nothing; Conductor's
    # 900-byte `PiecewiseLinearSpectrum{56}`-bearing struct collapses to a
    # 40-byte `ConductorEvaluated`, eliminating 8-16 redundant
    # `eval_ior_spectral` binary searches per surface hit.
    tfc_for_bxdf = compute_texture_filter_context(mat_work, camera, samples_per_pixel)
    regularize_for_bxdf = do_regularize && work.any_non_specular_bounces
    bxdf = get_bxdf(mat, rgb2spec_table, materials, tfc_for_bxdf, work.lambda, regularize_for_bxdf)

    # Direct lighting + inline shadow + accumulate (typed BSDF eval).
    surface_direct_lighting_inner_typed!(
        bxdf,
        pixel_L, accel, media_interfaces, media,
        mat_work, materials, lights, rgb2spec_table,
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights, num_lights,
        sobol_rng, sample_idx,
        camera, samples_per_pixel,
        do_regularize,
    )

    # BSDF sample + RR + push continuation (typed BSDF sample).
    evaluate_material_inner_typed!(
        bxdf,
        next_ray_queue,
        mat_work, materials, rgb2spec_table, max_depth,
        do_regularize,
        sobol_rng, sample_idx,
        camera, samples_per_pixel,
        rr_depth,
    )
    return
end

"""Drain `hit_area_light_queue`: applies emission-MIS to each surface hit
that landed on an area-light triangle. Direct port of pbrt-v4's
"Handle emitters hit by indirect rays" kernel
(wavefront/integrator.cpp:540-572). Runs in parallel with `vp_shade_typed!`
because both write only to per-pixel atomic `pixel_L` accumulators (the
emitter kernel writes only `pixel_L`; the material kernels also write
`next_ray_queue` but that's disjoint from `pixel_L`)."""
@propagate_inbounds function vp_handle_emitters_kernel!(
    work,                            # VPHitAreaLightWorkItem (per-thread)
    pixel_L,
    lights, rgb2spec_table,
    bvh_nodes, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
)
    # Le(p, n, uv, wo)
    light_idx = flat_to_light_index(lights, Int32(work.arealight_flat_idx))
    Le = with_index(arealight_Le, lights, light_idx,
        lights, rgb2spec_table, work.wo, Vec3f(work.n), work.uv, work.lambda,
    )
    is_black(Le) && return

    contribution = work.beta * Le
    final_contrib = if work.depth == Int32(0) || work.specular_bounce
        contribution / average(work.r_u)
    else
        lightChoicePDF = bvh_pmf(
            bvh_nodes, light_to_bit_trail,
            num_infinite_lights, num_bvh_lights,
            work.prev_intr_p, work.prev_intr_n, Int32(work.arealight_flat_idx),
        )
        cos_theta = abs(dot(work.n, work.wo))   # wo = -ray.d → abs(dot(n, -ray.d)) = abs(dot(n, ray.d))
        lightPDF = if cos_theta > 0f0 && work.triangle_area > 0f0
            pdf_li = (work.t_hit * work.t_hit) / (cos_theta * work.triangle_area)
            lightChoicePDF * pdf_li
        else
            0f0
        end
        r_l = work.r_l * lightPDF
        mis_denom = average(work.r_u + r_l)
        if mis_denom > 1f-10
            contribution / mis_denom
        else
            contribution / average(work.r_u)
        end
    end

    base_idx = (work.pixel_index - Int32(1)) * Int32(4)
    accumulate_spectrum!(pixel_L, base_idx, final_contrib)
    return
end

"""Drain the per-bounce `hit_area_light_queue` produced by
`enqueue_after_intersection!`. Matches pbrt-v4's split between
`hitAreaLightQueue` (this kernel) and `MaterialEvalQueue` (the
per-material `vp_shade_typed!` kernels)."""
function vp_handle_emitters!(state::VolPathState, lights)
    num_lights = state.num_lights
    num_lights < Int32(1) && return nothing
    foreach(vp_handle_emitters_kernel!,
        state.hit_area_light_queue,
        state.pixel_L,
        lights, state.rgb2spec_table,
        state.bvh_nodes, state.light_to_bit_trail,
        state.num_infinite_lights, state.num_bvh_lights, state.num_lights,
    )
    return nothing
end

"""Drain the per-material typed queues — one indirect dispatch per concrete
material type, each kernel monomorphised on a single `TypedHitRef{T}`.

Wrapped in `Lava.concurrent_indirect_group` so the per-type dispatches
share one fused multi-prepare + barrier and overlap on idle SMs.  The
per-type kernels write to atomically-claimed slots in the shared
`next_ray_queue` and to per-pixel atomic `pixel_L` accumulators, so
overlap is safe."""
function vp_shade_typed!(
    state::VolPathState, accel, media_interfaces, media,
    materials, lights,
    sample_idx::Int32,
    camera, samples_per_pixel::Int32,
    regularize::Bool = true,
)
    # Deferred indirect group: ONE fused multi-prepare for all per-type
    # dispatches, one shared barrier, then the dispatches overlapped.
    # Grouping the medium/escaped/emitters stages in here too was tried and
    # benchmarked worse (see the note in render!'s bounce loop).
    concurrent_indirect_group() do
        foreach_type(vp_shade_material_kernel!,
            state.per_material_queue,
            state.hit_surface_queue,
            next_ray_queue(state),
            state.pixel_L,
            accel,
            media_interfaces,
            media,
            materials,
            lights,
            state.rgb2spec_table,
            state.bvh_nodes,
            state.infinite_light_indices,
            state.light_to_bit_trail,
            state.num_infinite_lights,
            state.num_bvh_lights,
            state.num_lights,
            state.max_depth,
            regularize,
            state.sobol_rng, sample_idx,
            camera, samples_per_pixel,
            state.rr_depth,
        )
    end
    return nothing
end
