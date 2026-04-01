# ============================================================================
# Spectral Material Dispatch (type-stable dispatch over StaticMultiTypeSet)
# ============================================================================

"""
    sample_spectral_material(table, materials::StaticMultiTypeSet, idx, wo, ns, tfc, lambda, u, rng, regularize=false)

Type-stable dispatch for spectral BSDF sampling.
Returns SpectralBSDFSample from the appropriate material type.
"""
@propagate_inbounds function sample_spectral_material(
    table::RGBToSpectrumTable, materials::StaticMultiTypeSet,
    idx::SetKey,
    wo::Vec3f, ns::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, u::Point2f, rng::Float32,
    regularize::Bool = false
)
    return with_index(sample_bsdf_spectral, materials, idx, table, materials, wo, ns, tfc, lambda, u, rng, regularize)
end

"""
    evaluate_spectral_material(table, materials::StaticMultiTypeSet, idx, wo, wi, ns, tfc, lambda)

Type-stable dispatch for spectral BSDF evaluation.
Returns (f::SpectralRadiance, pdf::Float32).
"""
@propagate_inbounds function evaluate_spectral_material(
    table::RGBToSpectrumTable, materials::StaticMultiTypeSet,
    idx::SetKey,
    wo::Vec3f, wi::Vec3f, ns::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths
)
    return with_index(evaluate_bsdf_spectral, materials, idx, table, materials, wo, wi, ns, tfc, lambda)
end

"""
    russian_roulette_spectral(beta, r_u, eta_scale, depth, rr_sample, min_depth=1)

Apply Russian roulette for path termination. Follows pbrt-v4:
  rrBeta = beta * etaScale / r_u.Average()
  q = max(0, 1 - rrBeta.MaxComponentValue())
Returns (should_continue::Bool, new_beta::SpectralRadiance).
"""
@propagate_inbounds function russian_roulette_spectral(
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    eta_scale::Float32,
    depth::Int32,
    rr_sample::Float32,
    min_depth::Int32=Int32(1)
)
    if depth <= min_depth
        return (true, beta)
    end
    # pbrt-v4: rrBeta = beta * etaScale / r_u.Average()
    r_u_avg = average(r_u)
    rr_beta = if r_u_avg > 1f-10
        beta * eta_scale / r_u_avg
    else
        beta * eta_scale
    end
    max_comp = max_component(rr_beta)
    if max_comp >= 1f0
        return (true, beta)
    end
    q = 1f0 - max_comp
    if rr_sample < q
        return (false, beta)
    else
        return (true, beta / (1f0 - q))
    end
end

"""
    get_surface_alpha_dispatch(materials::StaticMultiTypeSet, idx::SetKey, uv::Point2f) -> Float32

Type-stable dispatch for evaluating surface alpha at a UV point.
Returns alpha ∈ [0, 1] where 0 = fully transparent, 1 = fully opaque.
"""
@propagate_inbounds function get_surface_alpha_dispatch(
    materials::StaticMultiTypeSet, idx::SetKey, uv::Point2f
)::Float32
    return with_index(get_surface_alpha, materials, idx, materials, uv)
end
