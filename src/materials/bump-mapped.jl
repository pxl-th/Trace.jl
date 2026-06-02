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
# differentials). Matches pbrt-v4 materials.h::BumpMap: `if (du == 0) du = .0005f`.
# This is sub-texel on Crown's 1024×1024 bump maps; the earlier 1/256 default
# was 4× the texel size and averaged out the high-frequency ornamental
# engraving on the gold dome (mitra_right_back conductor + bump).
const BUMP_DEFAULT_DELTA = 5f-4

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
    perturb_bump_frame(bump, materials, ns, dpdu, dpdv, dndu, dndv, ng, tfc)
        -> (ns_perturbed, dpdus_perturbed)

Implements pbrt-v4 materials.h:BumpMap exactly:
    dpdu' = dpdu + (∂h/∂u) * n + h * dndu
    dpdv' = dpdv + (∂h/∂v) * n + h * dndv

`dpdu` and `dpdv` are the UNNORMALIZED surface partial derivatives.
pbrt-v4 (shapes.h:959 `ss = isect.dpdu`) feeds the BumpMap formula the
raw geometric ∂p/∂u; the tilt of n_p depends on the ratio |dhdu|/|dpdu|.
Passing the unit-length shading tangent over-tilted n_p by 2-60× — that
was the bumped-gold leak into the crown interior pinned by
`shadow_bumpgold_dome_over_velvet`.

`ng` is the geometric (face) normal — pbrt-v4 `SetShadingGeometry` flips the
new bumped normal against `ng`, not against the original interpolated `ns`.
"""
@propagate_inbounds function perturb_bump_frame(bump, materials,
                                                ns::Vec3f,
                                                dpdu::Vec3f, dpdv::Vec3f,
                                                dndu::Vec3f, dndv::Vec3f,
                                                ng::Vec3f,
                                                tfc::TextureFilterContext)
    uv = tfc.uv
    δu = 0.5f0 * (abs(tfc.dudx) + abs(tfc.dudy))
    δu == 0f0 && (δu = BUMP_DEFAULT_DELTA)
    δv = 0.5f0 * (abs(tfc.dvdx) + abs(tfc.dvdy))
    δv == 0f0 && (δv = BUMP_DEFAULT_DELTA)
    h0 = _bump_height(bump, materials, uv)
    hu = _bump_height(bump, materials, Point2f(uv[1] + δu, uv[2]))
    hv = _bump_height(bump, materials, Point2f(uv[1], uv[2] + δv))
    dhdu = (hu - h0) / δu
    dhdv = (hv - h0) / δv

    # pbrt-v4 materials.h:BumpMap eq. 9.20 — uses unnormalized dpdu/dpdv directly
    dpdu_p = dpdu + dhdu * ns + h0 * dndu
    dpdv_p = dpdv + dhdv * ns + h0 * dndv

    # pbrt-v4 interaction.cpp:184-187 — `Normal3f ns(Normalize(Cross(dpdu, dpdv)))`
    # then `SetShadingGeometry(ns, dpdu, dpdv, ..., false)`, which inside
    # SetShadingGeometry runs `shading.n = FaceForward(shading.n, n)` — i.e.
    # flips the new bumped normal so it sits on the same side of the surface
    # as the GEOMETRIC normal (`n`), not the interpolated `shading.n`.
    n_p_raw = cross(dpdu_p, dpdv_p)
    n_len = sqrt(dot(n_p_raw, n_p_raw))
    n_p = n_len > 1f-10 ? n_p_raw / n_len : ns
    if dot(n_p, ng) < 0f0
        n_p = -n_p   # FaceForward(bumped_ns, geometric_n)
    end
    # Hikari's BSDFs build their local frame from (ns, dpdus) via
    # shading_frame, which Gram-Schmidts internally; pbrt-v4 does the same
    # in BSDFFrame::FromXZ. So we hand back the raw bumped tangent — no
    # extra orthonormalization, matching pbrt-v4 exactly.
    return n_p, dpdu_p
end

# ─────────────────────────────────────────────────────────────────────────────
# BSDF dispatch — pure passthrough to the inner material. The bump
# perturbation is now applied at intersection time by
# `get_perturbed_shading_frame` (see dispatch.jl). Doing it here as well
# would double-apply the height-field gradient on every sample/evaluate.
# ─────────────────────────────────────────────────────────────────────────────

@propagate_inbounds sample_bsdf_spectral(
    mat::BumpMapped, table::RGBToSpectrumTable, materials,
    wo::Vec3f, ns::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false,
) = sample_bsdf_spectral(mat.inner, table, materials, wo, ns, dpdus, tfc,
                         lambda, sample_u, rng, regularize)

@propagate_inbounds evaluate_bsdf_spectral(
    mat::BumpMapped, table::RGBToSpectrumTable, materials,
    wo::Vec3f, wi::Vec3f, ns::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, regularize::Bool = false,
) = evaluate_bsdf_spectral(mat.inner, table, materials, wo, wi, ns, dpdus, tfc,
                           lambda, regularize)
