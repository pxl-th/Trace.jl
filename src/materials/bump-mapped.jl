# ============================================================================
# BumpMapped — Material wrapper that perturbs the shading frame from a height
# texture, then delegates BSDF sampling/evaluation to the inner material.
#
# Used for pbrt-v4's `displacement` / `bumpmap` parameter (and indirectly for
# `normalmap`, which is just a different way of saying "use this texture to
# perturb the shading normal"). Without it, Hikari rendered scenes like Crown
# as smooth surfaces; with it, displacement textures on gold and pearl
# materials show through.
#
# Algorithm follows pbrt-v4 §9.3 "Bump Mapping":
#   1. Sample the height texture h at (u, v), (u+ε, v), (u, v+ε).
#   2. Compute ∂h/∂u, ∂h/∂v by finite differences.
#   3. Perturb the tangent vectors:
#        dpdu' = dpdu + (∂h/∂u) * ns
#        dpdv' = dpdv + (∂h/∂v) * ns
#   4. New shading normal n' = normalize(dpdu' × dpdv'); preserve sign.
#   5. Re-orthonormalize dpdu' against n'.
#
# `dpdvs` is reconstructed at evaluation time as cross(ns, dpdus), mirroring
# how `vp_compute_shading_tangents` builds it in volpath/intersection.jl.
# ============================================================================

# Floor on the finite-difference step when the texture-filter screen-space
# derivatives are zero (e.g. when the integrator does not propagate ray
# differentials). This is wider than pbrt's 5e-4 because Hikari's TextureFilter
# Context currently passes du/dv = 0; without ray differentials we need a
# step large enough to actually span one or two bump-texture texels, otherwise
# the finite difference returns ~zero gradient.
const BUMP_DEFAULT_DELTA = Float32(1 / 256)

"""
    BumpMapped(inner::Material, bump::Texture / TextureRef)

Wrap `inner` with a per-shading-point height-field normal perturbation.
"""
struct BumpMapped{M<:Material, T} <: Material
    inner::M
    bump::T
end
# (Julia synthesizes the parser-friendly outer constructor
#  `BumpMapped(::M, ::T)` automatically; no explicit one needed.)

# Forward the material-level traits to the inner material.
@propagate_inbounds is_emissive(mat::BumpMapped) = is_emissive(mat.inner)
@propagate_inbounds is_pure_emissive(mat::BumpMapped) = is_pure_emissive(mat.inner)
@propagate_inbounds get_emission(mat::BumpMapped, wo::Vec3f, n::Vec3f, uv::Point2f) =
    get_emission(mat.inner, wo, n, uv)
@propagate_inbounds get_emission(mat::BumpMapped, uv::Point2f) =
    get_emission(mat.inner, uv)
@propagate_inbounds get_surface_alpha(mat::BumpMapped, textures, uv::Point2f) =
    get_surface_alpha(mat.inner, textures, uv)

# ─────────────────────────────────────────────────────────────────────────────
# Height-field gradient → perturbed shading frame
# ─────────────────────────────────────────────────────────────────────────────

# Sample the height (float) at a uv. The bump texture is a float texture (the
# scalar height field); spectrum bump textures fall back to luminance via the
# float branch of imagemap loading in build_pbrt_textures.
@propagate_inbounds function _bump_height(bump, materials, uv::Point2f)
    h = eval_tex(materials, bump, uv)
    return h isa Real ? Float32(h) : Float32(h.c[1])
end

"""
    perturb_bump_frame(bump, materials, ns, dpdus, tfc) -> (ns_perturbed, dpdus_perturbed)

Return a perturbed shading normal and tangent from height-field gradient.
"""
@propagate_inbounds function perturb_bump_frame(bump, materials,
                                                ns::Vec3f, dpdus::Vec3f,
                                                tfc::TextureFilterContext)
    uv = tfc.uv
    # pbrt-v4-style derivative-driven step: average the texture-filter screen-
    # space (u,v) derivatives, falling back to BUMP_DEFAULT_DELTA when those
    # are zero. Forward differences (rather than central) match pbrt's
    # BumpMapping; the rest of the routine reproduces equations 9.20-9.22.
    δu = 0.5f0 * (abs(tfc.dudx) + abs(tfc.dudy))
    δu == 0f0 && (δu = BUMP_DEFAULT_DELTA)
    δv = 0.5f0 * (abs(tfc.dvdx) + abs(tfc.dvdy))
    δv == 0f0 && (δv = BUMP_DEFAULT_DELTA)
    h0 = _bump_height(bump, materials, uv)
    hu = _bump_height(bump, materials, Point2f(uv[1] + δu, uv[2]))
    hv = _bump_height(bump, materials, Point2f(uv[1], uv[2] + δv))
    dhdu = (hu - h0) / δu
    dhdv = (hv - h0) / δv

    # Reconstruct the binormal the same way the volpath integrator does.
    dpdvs = cross(ns, dpdus)

    dpdu_p = dpdus + dhdu * ns
    dpdv_p = dpdvs + dhdv * ns

    n_p_raw = cross(dpdu_p, dpdv_p)
    n_len = sqrt(dot(n_p_raw, n_p_raw))
    n_p = n_len > 0f0 ? n_p_raw / n_len : ns
    # Preserve original normal hemisphere (matches pbrt-v4's FaceForward).
    if dot(n_p, ns) < 0f0
        n_p = -n_p
    end

    # Re-orthonormalize the tangent against the new normal so dpdus stays
    # perpendicular to ns (the convention every Hikari BSDF expects).
    dpdus_p = dpdu_p - n_p * dot(n_p, dpdu_p)
    len_t = sqrt(dot(dpdus_p, dpdus_p))
    dpdus_out = len_t > 1f-8 ? dpdus_p / len_t : dpdus

    return n_p, dpdus_out
end

# ─────────────────────────────────────────────────────────────────────────────
# BSDF dispatch overrides — perturb the frame, then delegate
# ─────────────────────────────────────────────────────────────────────────────

@propagate_inbounds function sample_bsdf_spectral(
    mat::BumpMapped, table::RGBToSpectrumTable, materials,
    wo::Vec3f, ns::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    ns_p, dpdus_p = perturb_bump_frame(mat.bump, materials, ns, dpdus, tfc)
    return sample_bsdf_spectral(mat.inner, table, materials,
                                wo, ns_p, dpdus_p, tfc,
                                lambda, sample_u, rng, regularize)
end

@propagate_inbounds function evaluate_bsdf_spectral(
    mat::BumpMapped, table::RGBToSpectrumTable, materials,
    wo::Vec3f, wi::Vec3f, ns::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, regularize::Bool = false
)
    ns_p, dpdus_p = perturb_bump_frame(mat.bump, materials, ns, dpdus, tfc)
    return evaluate_bsdf_spectral(mat.inner, table, materials,
                                  wo, wi, ns_p, dpdus_p, tfc,
                                  lambda, regularize)
end
