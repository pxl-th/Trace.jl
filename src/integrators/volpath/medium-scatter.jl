# Medium scattering kernel for volumetric path tracing
# Handles real scattering events: direct lighting + phase function sampling

# ============================================================================
# Medium Direct Lighting Inner Function
# ============================================================================

"""Inner function for medium direct lighting - can use return statements.

Uses power-weighted light sampling via alias table for better importance sampling
in scenes with lights of varying intensities (pbrt-v4's PowerLightSampler approach).

Sobol samples are generated inline rather than read from a pre-populated
per-pixel buffer; see `surface_direct_lighting_inner!` in surface-eval.jl
for the rationale.
"""
@propagate_inbounds function medium_direct_lighting_inner!(
    shadow_queue,
    work::VPMediumScatterWorkItem,
    lights,
    rgb2spec_table,
    bvh_nodes,
    infinite_light_indices,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    num_lights::Int32,
    sobol_rng,           # SobolRNG — samples generated on demand inline
    sample_idx::Int32,
)
    # Skip if no lights
    num_lights < Int32(1) && return

    # Inline Sobol sample generation (matches dimension layout used by the
    # surface path's `surface_direct_lighting_inner!`).
    pixel_idx = work.pixel_index
    pixel_idx_0 = pixel_idx - Int32(1)
    px = u_mod(pixel_idx_0, sobol_rng.width) + Int32(1)
    py = u_div(pixel_idx_0, sobol_rng.width) + Int32(1)
    base_dim = Int32(6) + Int32(7) * work.depth
    light_select = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(1))
    u_light_x, u_light_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(3))
    u_light = Point2f(u_light_x, u_light_y)

    # Select light using BVH light sampler (spatially-aware importance sampling)
    # Medium scattering has no surface normal → pass Vec3f(0f0)
    light_idx, light_pmf = bvh_sample_light(
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights,
        work.p, Vec3f(0f0), light_select
    )

    # Validate index
    if light_idx < Int32(1) || light_idx > num_lights || light_pmf <= 0f0
        return
    end

    # Sample the light (works with both Tuple and StaticMultiTypeSet)
    light_sample = sample_light_spectral(
        rgb2spec_table, lights, light_idx, work.p, work.lambda, u_light
    )

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        # Evaluate phase function for light direction
        cos_θ = dot(work.wo, light_sample.wi)
        phase_val = hg_p(work.g, cos_θ)

        if phase_val > 0f0
            # Compute direct lighting contribution following pbrt-v4 (media.cpp lines 287-300):
            # Ld = beta * phase * Li
            # NO PDF division here - that happens at shadow ray resolution via MIS weights
            Ld = work.beta * phase_val * light_sample.Li

            # MIS weights following pbrt-v4 (media.cpp lines 293-297):
            # lightPDF = ls->pdf * sampledLight->p (light PDF including selection probability)
            # phasePDF = 0 for delta lights, else phase->PDF(wo, wi)
            # r_u = w.r_u * phasePDF
            # r_l = w.r_u * lightPDF
            # Now using power-weighted PMF instead of uniform 1/num_lights
            light_pdf = light_sample.pdf * light_pmf  # Include light selection probability
            phase_pdf = if light_sample.is_delta
                0f0  # Delta lights have no MIS with phase sampling
            else
                phase_val  # HG PDF equals value for importance sampling
            end
            r_u = work.r_u * phase_pdf
            r_l = work.r_u * light_pdf

            # Shadow ray from scatter point to light
            shadow_origin = work.p
            shadow_dir = light_sample.wi
            t_max = if light_sample.is_delta
                # Point/directional light - go to light position
                norm(light_sample.p_light - work.p) - 0.001f0
            else
                # Area light - use large distance
                1f6
            end

            shadow_ray = Raycore.Ray(
                o = shadow_origin,
                d = shadow_dir,
                t_max = t_max,
                time = work.time
            )

            shadow_item = VPShadowRayWorkItem(
                shadow_ray,
                t_max,
                work.lambda,
                Ld,
                r_u,
                r_l,
                work.pixel_index,
                work.medium_idx  # Shadow ray travels through same medium
            )

            push!(shadow_queue, shadow_item)
        end
    end
    return
end

# ============================================================================
# Medium Scatter Kernel (Direct Lighting)
# ============================================================================

@propagate_inbounds function vp_medium_direct_lighting_kernel!(
    work,
    shadow_queue,
    lights,
    rgb2spec_table,
    bvh_nodes, infinite_light_indices,
    num_infinite_lights::Int32, num_bvh_lights::Int32,
    num_lights::Int32,
    sobol_rng, sample_idx_ref,
)
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    medium_direct_lighting_inner!(
        shadow_queue,
        work, lights, rgb2spec_table,
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights,
        num_lights,
        sobol_rng, sample_idx,
    )
end

# ============================================================================
# Medium Scatter Inner Function
# ============================================================================

"""Inner function for medium scatter - can use return statements.

Sobol samples are generated inline; see `surface_direct_lighting_inner!`.
"""
@propagate_inbounds function medium_scatter_inner!(
    ray_queue,
    work::VPMediumScatterWorkItem,
    max_depth::Int32,
    sobol_rng,
    sample_idx::Int32,
)
    # Check depth limit
    new_depth = work.depth + Int32(1)
    if new_depth >= max_depth
        return
    end

    # Inline Sobol sample generation for phase function direction (dim+6 = 2D).
    pixel_idx = work.pixel_index
    pixel_idx_0 = pixel_idx - Int32(1)
    px = u_mod(pixel_idx_0, sobol_rng.width) + Int32(1)
    py = u_div(pixel_idx_0, sobol_rng.width) + Int32(1)
    base_dim = Int32(6) + Int32(7) * work.depth
    u_x, u_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(6))
    u = Point2f(u_x, u_y)
    wi, phase_pdf = sample_hg(work.g, work.wo, u)

    if phase_pdf > 0f0
        # Update throughput
        # For phase functions: f = phase, pdf = phase, so ratio = 1
        # But we need f/pdf for path throughput
        new_beta = work.beta  # phase/pdf = 1 for importance-sampled HG

        # Update MIS weights
        new_r_u = work.r_u
        new_r_l = work.r_u / phase_pdf

        # Create continuation ray
        new_ray = Raycore.Ray(
            o = work.p,
            d = wi,
            t_max = Inf32,
            time = work.time
        )

        ray_item = VPRayWorkItem(
            new_ray,
            new_depth,
            work.lambda,
            work.pixel_index,
            new_beta,
            new_r_u,
            new_r_l,
            work.p,           # prev_intr_p
            work.wo,          # prev_intr_n (use wo as pseudo-normal for MIS)
            work.eta_scale,   # eta_scale (carry through from path state)
            false,            # specular_bounce
            true,             # any_non_specular_bounces
            work.medium_idx   # Stay in same medium
        )

        push!(ray_queue, ray_item)
    end
    return
end

# ============================================================================
# Medium Scatter Kernel (Phase Function Sampling)
# ============================================================================

@propagate_inbounds function vp_medium_scatter_kernel!(
    work,
    ray_queue,
    max_depth::Int32,
    sobol_rng, sample_idx_ref,
)
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    medium_scatter_inner!(ray_queue, work, max_depth, sobol_rng, sample_idx)
end

