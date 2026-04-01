# ============================================================================
# Dielectric (Glass) and ThinDielectric Materials
# Contains: Dielectric struct, keyword constructor, Dielectric alias,
#           ThinDielectric struct, keyword constructor, ThinDielectric alias,
#           eval_dielectric_ior helpers,
#           sample_bsdf_spectral and evaluate_bsdf_spectral methods for both
# ============================================================================

"""
    Dielectric(Kr, Kt, u_roughness, v_roughness, index, remap_roughness)

Glass/dielectric material with reflection and transmission.

* `Kr`: Spectral reflectance (Texture or TextureRef)
* `Kt`: Spectral transmittance (Texture or TextureRef)
* `u_roughness`: Roughness in u direction (0 = perfect specular)
* `v_roughness`: Roughness in v direction (0 = perfect specular)
* `index`: Index of refraction
* `remap_roughness`: Whether to remap roughness to alpha
"""
struct Dielectric{KrTex, KtTex, URoughTex, VRoughTex, IndexTex} <: Material
    Kr::KrTex           # Texture, Raycore.TextureRef, or raw RGBSpectrum
    Kt::KtTex           # Texture, Raycore.TextureRef, or raw RGBSpectrum
    u_roughness::URoughTex  # Texture, Raycore.TextureRef, or raw Float32
    v_roughness::VRoughTex  # Texture, Raycore.TextureRef, or raw Float32
    index::IndexTex     # Texture, Raycore.TextureRef, or raw Float32
    remap_roughness::Bool
end

"""
    Dielectric(; Kr=RGBSpectrum(1), Kt=RGBSpectrum(1), roughness=0, index=1.5, remap_roughness=true)

Create a glass/dielectric material with reflection and transmission.

# Arguments
- `Kr`: Reflectance color
- `Kt`: Transmittance color
- `roughness`: Surface roughness (0 = perfect specular, can be single value or (u,v) tuple)
- `index`: Index of refraction (1.5 for glass, 1.33 for water, 2.4 for diamond)
- `remap_roughness`: Whether to remap roughness to microfacet alpha

# Examples
```julia
Dielectric()                                # Clear glass
Dielectric(Kt=(1, 0.9, 0.8), index=1.5)    # Amber tinted
Dielectric(roughness=0.1)                   # Frosted glass
Dielectric(roughness=(0.1, 0.05))          # Anisotropic roughness
```
"""
function Dielectric(;
    Kr=RGBSpectrum(1f0),
    Kt=RGBSpectrum(1f0),
    roughness=0f0,
    index=1.5f0,
    remap_roughness=true
)
    # Handle roughness - can be single value or (u, v) tuple
    if roughness isa Tuple
        u_rough, v_rough = roughness
    else
        u_rough = v_rough = roughness
    end
    Dielectric(
        to_texture(Kr), to_texture(Kt),
        to_texture(u_rough), to_texture(v_rough),
        to_texture(index), remap_roughness
    )
end


# ============================================================================
# ThinDielectric - Thin dielectric surface (e.g., window glass)
# ============================================================================
# Port of pbrt-v4's ThinDielectric and ThinDielectricBxDF
#
# Models a thin dielectric surface where light can pass through without
# the usual refraction bend. This is appropriate for thin surfaces like
# window glass where internal bounces contribute to the overall reflection.
#
# Key difference from regular Dielectric:
# - Transmitted direction is -wo (straight through, no refraction)
# - Reflectance accounts for multiple internal bounces: R' = R + T²R/(1-R²)
# - Always specular (no roughness parameter)
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h ThinDielectricBxDF (lines 209-277)

"""
    ThinDielectric

A thin dielectric material for surfaces like window glass.

Unlike regular dielectric materials which refract light according to Snell's law,
thin dielectric materials transmit light straight through (wi = -wo) while
accounting for multiple internal reflections within the thin layer.

# Fields
- `eta`: Index of refraction of the dielectric

# Physics
For a thin dielectric layer:
- Single-surface Fresnel: R₀ = FrDielectric(cos_θ, eta)
- Multiple-bounce reflectance: R = R₀ + T₀²R₀/(1 - R₀²) where T₀ = 1 - R₀
- Transmittance: T = 1 - R
- Transmitted direction: wi = -wo (straight through, no bend)

# Usage
```julia
# Thin glass window
window = ThinDielectric(eta=1.5)

# Thin plastic film
film = ThinDielectric(eta=1.4)
```
"""
struct ThinDielectric{E} <: Material
    eta::E  # Float32 or PiecewiseLinearSpectrum
end

# Keyword constructor
function ThinDielectric(; eta=1.5f0)
    if eta isa PiecewiseLinearSpectrum
        ThinDielectric(eta)
    else
        ThinDielectric(Float32(eta))
    end
end

# Mark as non-emissive
is_emissive(::ThinDielectric) = false


# ============================================================================
# Helpers for dielectric IOR evaluation
# ============================================================================

# Evaluate dielectric IOR: returns (ior::Float32, is_dispersive::Bool)
# PiecewiseLinearSpectrum → sample at hero wavelength only, dispersive
@inline eval_dielectric_ior(textures, idx::PiecewiseLinearSpectrum, tfc, lambda) =
    (sample(idx, lambda.lambda[1]), true)
# Scalar/texture IOR → not dispersive
@inline function eval_dielectric_ior(textures, idx, tfc, lambda)
    ior = eval_tex(textures, idx, tfc)
    return (ior, false)
end

# ============================================================================
# Dielectric spectral BSDF sampling
# ============================================================================

"""
    sample_bsdf_spectral(table, mat::Dielectric, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample glass BSDF with reflection or refraction.
Uses Fresnel to choose between reflection and transmission.
"""
@propagate_inbounds function sample_bsdf_spectral(
    mat::Dielectric, table::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Get material properties
    kr_rgb = eval_tex(textures, mat.Kr, tfc)
    kt_rgb = eval_tex(textures, mat.Kt, tfc)

    # Evaluate IOR — pbrt-v4 DielectricMaterial::GetBxDF:
    # Float sampledEta = eta(lambda[0]);
    # if (!eta.Is<ConstantSpectrum>()) lambda.TerminateSecondary();
    ior, is_dispersive = eval_dielectric_ior(textures, mat.index, tfc, lambda)

    # Handle edge case where IOR is 0 (matches pbrt-v4 DielectricMaterial)
    ior == 0f0 && (ior = 1f0)

    kr_spectral = uplift_rgb(table, kr_rgb, lambda)
    kt_spectral = uplift_rgb(table, kt_rgb, lambda)

    # Get roughness and compute alpha
    u_roughness = eval_tex(textures, mat.u_roughness, tfc)
    v_roughness = eval_tex(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness

    # Apply regularization if requested
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end

    # Determine if entering or exiting
    cos_theta_o = dot(wo, n)
    entering = cos_theta_o > 0f0

    n_oriented = entering ? n : -n
    cos_theta_o = abs(cos_theta_o)

    # pbrt-v4 convention: eta = n_t / n_i
    eta = entering ? ior : (1f0 / ior)

    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y) || eta == 1f0

    if is_smooth
        # === Specular case (smooth dielectric) ===
        R = fresnel_dielectric(cos_theta_o, eta)
        T = 1f0 - R

        if rng < R
            wi = reflect(wo, n_oriented)
            f_val = kr_spectral * (R / cos_theta_o)
            return SpectralBSDFSample(f_val, wi, R, BXDF_SPECULAR_REFLECTION, 1f0, false, is_dispersive)
        else
            sin2_theta_i = max(0f0, 1f0 - cos_theta_o * cos_theta_o)
            sin2_theta_t = sin2_theta_i / (eta * eta)
            if sin2_theta_t >= 1f0
                wi = reflect(wo, n_oriented)
                f_val = kr_spectral * (1f0 / cos_theta_o)
                return SpectralBSDFSample(f_val, wi, 1f0, BXDF_SPECULAR_REFLECTION, 1f0, false, is_dispersive)
            end
            cos_theta_t = sqrt(1f0 - sin2_theta_t)
            wi = normalize(-wo / eta + (cos_theta_o / eta - cos_theta_t) * n_oriented)
            etap = eta
            f_val = kt_spectral * (T / cos_theta_t / (etap * etap))
            return SpectralBSDFSample(f_val, wi, T, BXDF_SPECULAR_TRANSMISSION, etap, false, is_dispersive)
        end
    else
        # === Rough case (microfacet dielectric) ===
        # Delegate to sample_dielectric_interface (same code used by CoatedDiffuse/CoatedConductor)
        # which is proven correct at all roughness levels.
        tangent, bitangent = shading_frame(n_oriented, dpdus)
        wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, n_oriented))

        bs = sample_dielectric_interface(wo_local, rng, sample_u, alpha_x, alpha_y, eta, BXDF_ALL)
        if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
            return SpectralBSDFSample()
        end

        # Transform wi back to world space
        wi = tangent * bs.wi[1] + bitangent * bs.wi[2] + n_oriented * bs.wi[3]
        wi = normalize(wi)

        # Apply spectral color
        f_val = bs.is_reflection ? kr_spectral * bs.f : kt_spectral * bs.f
        flags = if bs.is_reflection
            bs.is_specular ? BXDF_SPECULAR_REFLECTION : BXDF_GLOSSY_REFLECTION
        else
            bs.is_specular ? BXDF_SPECULAR_TRANSMISSION : BXDF_GLOSSY_TRANSMISSION
        end
        return SpectralBSDFSample(f_val, wi, bs.pdf, flags, bs.eta, false, is_dispersive)
    end
end

# ============================================================================
# Dielectric spectral BSDF evaluation (for MIS in direct lighting)
# ============================================================================

@propagate_inbounds function evaluate_bsdf_spectral(
    mat::Dielectric, table::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths
)
    ior, _ = eval_dielectric_ior(textures, mat.index, tfc, lambda)
    ior == 0f0 && (ior = 1f0)

    u_roughness = eval_tex(textures, mat.u_roughness, tfc)
    v_roughness = eval_tex(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness

    if trowbridge_reitz_effectively_smooth(alpha_x, alpha_y) || ior == 1f0
        # Specular: zero for non-delta directions
        return (SpectralRadiance(), 0f0)
    end

    # Delegate to eval_dielectric_interface (same code used by CoatedDiffuse/CoatedConductor)
    kr_rgb = eval_tex(textures, mat.Kr, tfc)
    kt_rgb = eval_tex(textures, mat.Kt, tfc)
    kr_spectral = uplift_rgb(table, kr_rgb, lambda)
    kt_spectral = uplift_rgb(table, kt_rgb, lambda)

    cos_theta_o = dot(wo, n)
    entering = cos_theta_o > 0f0
    n_oriented = entering ? n : -n
    eta = entering ? ior : (1f0 / ior)

    tangent, bitangent = shading_frame(n_oriented, dpdus)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, n_oriented))
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), dot(wi, n_oriented))

    f_raw, pdf = eval_dielectric_interface(wo_local, wi_local, alpha_x, alpha_y, eta)

    # Apply spectral color
    is_reflect = wo_local[3] * wi_local[3] > 0f0
    f_val = is_reflect ? kr_spectral * f_raw : kt_spectral * f_raw
    return (f_val, pdf)
end

# ============================================================================
# ThinDielectric spectral BSDF sampling
# ============================================================================

"""
    sample_bsdf_spectral(table, mat::ThinDielectric, textures, wo, n, uv, lambda, sample_u, rng, regularize=false) -> SpectralBSDFSample

Sample thin dielectric BSDF matching pbrt-v4's ThinDielectricBxDF::Sample_f.

Thin dielectric surfaces model materials like window glass where light can
either reflect or transmit straight through (no refraction bend).

Key physics (pbrt-v4 lines 225-230):
- R₀ = FrDielectric(|cos_θ|, eta)
- R = R₀ + T₀²R₀/(1 - R₀²)  where T₀ = 1 - R₀
- T = 1 - R
- Transmitted direction: wi = -wo (straight through)
- Reflected direction: wi = (-wo.x, -wo.y, wo.z) (mirror reflection in local coords)
"""
@propagate_inbounds function sample_bsdf_spectral(
    mat::ThinDielectric, table::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Evaluate IOR — pbrt-v4: sample at hero wavelength, terminate secondaries if dispersive
    eta, is_dispersive = eval_dielectric_ior(textures, mat.eta, tfc, lambda)

    # Build local coordinate frame
    tangent, bitangent = shading_frame(n, dpdus)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    # Compute single-interface Fresnel reflectance
    cos_θo = abs(wo_local[3])
    R0 = fresnel_dielectric(cos_θo, eta)
    T0 = 1f0 - R0

    # Account for multiple internal bounces (pbrt-v4 lines 227-230)
    # R = R0 + T0² * R0 / (1 - R0²)
    R = R0
    if R0 < 1f0
        R = R0 + T0 * T0 * R0 / (1f0 - R0 * R0)
    end
    T = 1f0 - R

    # Choose reflection or transmission based on rng
    pr = R
    pt = T

    if pr + pt < 1f-10
        return SpectralBSDFSample()
    end

    prob_reflect = pr / (pr + pt)

    if rng < prob_reflect
        # Sample perfect specular reflection
        # wi = (-wo.x, -wo.y, wo.z) in local coords
        wi_local = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])

        # Transform back to world space
        wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
        wi = normalize(wi)

        # f = R / |cos_θ|
        f_val = R / abs(wi_local[3])
        return SpectralBSDFSample(SpectralRadiance(f_val), wi, prob_reflect, BXDF_SPECULAR_REFLECTION, 1f0)
    else
        # Sample perfect specular transmission
        # wi = -wo (straight through, no refraction)
        wi = -wo

        # f = T / |cos_θ|
        f_val = T / cos_θo
        return SpectralBSDFSample(SpectralRadiance(f_val), wi, 1f0 - prob_reflect, BXDF_SPECULAR_TRANSMISSION, 1f0)
    end
end

# ============================================================================
# ThinDielectric spectral BSDF evaluation (for MIS in direct lighting)
# ============================================================================

"""
    evaluate_bsdf_spectral(table, mat::ThinDielectric, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate thin dielectric BSDF - returns zero for non-delta directions.
ThinDielectric is purely specular, so f() and PDF() both return 0.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    mat::ThinDielectric, table::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths
)
    # ThinDielectric is purely specular - f() returns 0 for all non-delta directions
    return (SpectralRadiance(), 0f0)
end
