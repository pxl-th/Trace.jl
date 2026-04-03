# ============================================================================
# Conductor (Metal) and Mirror Materials
# Contains: Mirror struct, keyword constructor, Mirror alias,
#           Conductor struct, keyword constructor, Conductor/Metal aliases,
#           Gold/Silver/Copper/Aluminum/Brass preset constructors,
#           eval_ior_spectral helpers,
#           sample_bsdf_spectral and evaluate_bsdf_spectral methods for both
# ============================================================================

# ============================================================================
# Mirror
# ============================================================================

"""
    Mirror(Kr::Texture)

Perfect mirror (specular reflection) material.

* `Kr`: Spectral reflectance (color texture or TextureRef)
"""
struct Mirror{KrTex} <: Material
    Kr::KrTex   # Texture, Raycore.TextureRef, or raw RGBSpectrum
end

"""
    Mirror(; Kr=RGBSpectrum(0.9))

Create a perfect mirror (specular reflection) material.

# Arguments
- `Kr`: Reflectance color - can be RGBSpectrum, (r,g,b) tuple, or Texture

# Examples
```julia
Mirror()                               # Default silver mirror
Mirror(Kr=RGBSpectrum(0.95, 0.93, 0.88))  # Gold-tinted
Mirror(Kr=(0.9, 0.9, 0.9))            # Using tuple
```
"""
function Mirror(; Kr=RGBSpectrum(0.9f0))
    Mirror(to_texture(Kr))
end


# ============================================================================
# Conductor
# ============================================================================

"""
    Conductor{EtaTex, KTex, RoughTex, ReflTex}

A metal/conductor material with wavelength-dependent complex index of refraction.

Metals reflect light based on Fresnel equations for conductors, characterized by:
- η (eta): Real part of complex IOR (PiecewiseLinearSpectrum for presets, or RGB texture)
- k: Imaginary part (extinction coefficient)
- roughness: Surface roughness for microfacet model

# Fields
* `eta`: Real part of complex index of refraction (PiecewiseLinearSpectrum or texture)
* `k`: Extinction coefficient (PiecewiseLinearSpectrum or texture)
* `roughness`: Surface roughness
* `reflectance`: Color multiplier for Fresnel reflectance (for tinting)
* `remap_roughness`: Whether to remap roughness to alpha
"""
struct Conductor{EtaTex, KTex, RoughTex, ReflTex} <: Material
    eta::EtaTex             # PiecewiseLinearSpectrum, Texture, Raycore.TextureRef, or raw RGBSpectrum
    k::KTex                 # PiecewiseLinearSpectrum, Texture, Raycore.TextureRef, or raw RGBSpectrum
    roughness::RoughTex     # Texture, Raycore.TextureRef, or raw Float32
    reflectance::ReflTex    # Texture, Raycore.TextureRef, or raw RGBSpectrum
    remap_roughness::Bool
end


"""
    Conductor(; eta=(0.2, 0.2, 0.2), k=(3.9, 3.9, 3.9), roughness=0.1, remap_roughness=true)

Create a metal/conductor material with Fresnel reflectance.

# Arguments
- `eta`: Real part of complex IOR - PiecewiseLinearSpectrum, (r,g,b) tuple, RGBSpectrum, or Texture
- `k`: Extinction coefficient - PiecewiseLinearSpectrum, (r,g,b) tuple, RGBSpectrum, or Texture
- `roughness`: Surface roughness (0 = mirror-like, higher = more diffuse)
- `reflectance`: Color multiplier for tinting the metal (default white = no tint)
- `remap_roughness`: Whether to remap roughness to microfacet alpha

# Presets
Use the spectral preset constructors: `Gold()`, `Silver()`, `Copper()`, `Aluminum()`, `Brass()`

# Examples
```julia
Conductor()                                        # Generic metal
Gold(roughness=0.05)                               # Polished gold
Conductor(eta=(0.2, 0.8, 0.2), k=(3, 3, 3))       # Custom green-tinted metal
```
"""
function Conductor(;
    eta=(0.2f0, 0.2f0, 0.2f0),
    k=(3.9f0, 3.9f0, 3.9f0),
    roughness=0.1f0,
    reflectance=(1f0, 1f0, 1f0),
    remap_roughness=true
)
    Conductor(to_texture(eta), to_texture(k), to_texture(roughness), to_texture(reflectance), remap_roughness)
end

# ============================================================================
# Conductor Preset Constructors - using measured spectral data from pbrt-v4
# ============================================================================

"""
    Gold(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a gold conductor material with measured spectral IOR data.

# Examples
```julia
Gold()                          # Polished gold
Gold(roughness=0.1)             # Brushed gold
Gold(roughness=0.3)             # Matte gold
```
"""
Gold(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(AU_ETA_SPECTRUM, AU_K_SPECTRUM, to_texture(roughness), to_texture(reflectance), remap_roughness)

"""
    Silver(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a silver conductor material with measured spectral IOR data.

# Examples
```julia
Silver()                        # Polished silver
Silver(roughness=0.05)          # Slightly brushed
```
"""
Silver(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(AG_ETA_SPECTRUM, AG_K_SPECTRUM, to_texture(roughness), to_texture(reflectance), remap_roughness)

"""
    Copper(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a copper conductor material with measured spectral IOR data.

# Examples
```julia
Copper()                        # Polished copper
Copper(roughness=0.2)           # Weathered copper
```
"""
Copper(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(CU_ETA_SPECTRUM, CU_K_SPECTRUM, to_texture(roughness), to_texture(reflectance), remap_roughness)

"""
    Aluminum(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create an aluminum conductor material with measured spectral IOR data.

# Examples
```julia
Aluminum()                      # Polished aluminum
Aluminum(roughness=0.1)         # Brushed aluminum
```
"""
Aluminum(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(AL_ETA_SPECTRUM, AL_K_SPECTRUM, to_texture(roughness), to_texture(reflectance), remap_roughness)

"""
    Brass(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a brass (CuZn) conductor material with measured spectral IOR data.

# Examples
```julia
Brass()                         # Polished brass
Brass(roughness=0.15)           # Brushed brass
```
"""
Brass(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(CUZN_ETA_SPECTRUM, CUZN_K_SPECTRUM, to_texture(roughness), to_texture(reflectance), remap_roughness)

# ============================================================================
# Helpers for evaluating IOR values (PiecewiseLinearSpectrum or RGB textures)
# ============================================================================

# ============================================================================
# Mirror spectral BSDF sampling
# ============================================================================

"""
    sample_bsdf_spectral(table, mat::Mirror, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample perfect specular reflection with spectral evaluation.
"""
@propagate_inbounds function sample_bsdf_spectral(
    mat::Mirror, table::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get reflectance (rng and sample_u unused for perfect specular)
    kr_rgb = eval_tex(textures, mat.Kr, tfc)
    kr_spectral = uplift_rgb(table, kr_rgb, lambda)

    # Orient normal to face wo for reflection
    n_oriented = wo_dot_n < 0f0 ? -n : n

    # Perfect reflection
    wi = reflect(wo, n_oriented)

    # Delta distribution: f = Kr / cos_theta, pdf = 1
    cos_theta_i = abs(dot(wi, n_oriented))
    return SpectralBSDFSample(kr_spectral / cos_theta_i, wi, 1f0, BXDF_SPECULAR_REFLECTION, 1f0)
end

# ============================================================================
# Mirror spectral BSDF evaluation (for MIS in direct lighting)
# ============================================================================

@propagate_inbounds function evaluate_bsdf_spectral(
    mat::Mirror, table::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths,
    regularize::Bool = false
)
    # Perfect specular has zero PDF for non-delta directions
    return (SpectralRadiance(), 0f0)
end

# ============================================================================
# Conductor spectral BSDF sampling
# ============================================================================

"""
    sample_bsdf_spectral(table, mat::Conductor, textures, wo, n, uv, lambda, sample_u, rng, regularize=false) -> SpectralBSDFSample

Sample metal BSDF with conductor Fresnel.
Matches pbrt-v4's ConductorBxDF::Sample_f exactly.

The implementation works in local shading coordinates where n = (0,0,1), then transforms back.

When `regularize=true`, the microfacet alpha is increased to reduce fireflies
from near-specular paths (matches pbrt-v4 BSDF::Regularize).
"""
@propagate_inbounds function sample_bsdf_spectral(
    mat::Conductor, table::RGBToSpectrumTable, textures,
    wo_world::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Build local coordinate frame (matches pbrt-v4's BSDF shading frame)
    tangent, bitangent = shading_frame(n, dpdus)

    # Transform wo to local coordinates (matches pbrt-v4's RenderToLocal)
    wo = world_to_local(wo_world, n, tangent, bitangent)

    # Check for grazing angle (matches pbrt-v4's wo.z == 0 check)
    if wo[3] == 0f0
        return SpectralBSDFSample()
    end

    # Get material properties
    roughness = eval_tex(textures, mat.roughness, tfc)

    # Compute alpha values (matches pbrt-v4's roughness remapping)
    alpha_x = mat.remap_roughness ? roughness_to_α(roughness) : roughness
    alpha_y = alpha_x  # Isotropic for now

    # Apply regularization if requested (pbrt-v4: doubles alpha if < 0.3, clamps to [0.1, 0.3])
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end

    # Clamp alpha to minimum value if not smooth (matches pbrt-v4 TrowbridgeReitzDistribution constructor)
    if !trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        alpha_x = max(alpha_x, 1f-4)
        alpha_y = max(alpha_y, 1f-4)
    end

    # Evaluate eta and k spectrally (dispatches on PiecewiseLinearSpectrum vs RGB texture)
    eta_spectral = eval_ior_spectral(table, textures, mat.eta, tfc, lambda)
    k_spectral = eval_ior_spectral(table, textures, mat.k, tfc, lambda)

    if trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        # Sample perfect specular conductor BRDF (matches pbrt-v4 line 301-305)
        # wi = (-wo.x, -wo.y, wo.z) in local coordinates
        wi = Vec3f(-wo[1], -wo[2], wo[3])

        # f = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi)
        cos_theta_i = abs_cos_theta(wi)
        F = fr_complex_spectral(cos_theta_i, eta_spectral, k_spectral)
        f = F / cos_theta_i

        # Transform wi back to world coordinates
        wi_world = local_to_world(wi, n, tangent, bitangent)

        return SpectralBSDFSample(f, wi_world, 1f0, BXDF_SPECULAR_REFLECTION, 1f0)
    else
        # Sample rough conductor BRDF (matches pbrt-v4 line 307-327)

        # Sample microfacet normal wm (matches pbrt-v4 line 311)
        wm = trowbridge_reitz_sample_wm(wo, sample_u, alpha_x, alpha_y)

        # Compute reflected direction (matches pbrt-v4 line 312)
        # Reflect(wo, wm) = -wo + 2 * dot(wo, wm) * wm
        wi = -wo + 2f0 * dot(wo, wm) * wm

        # Reject if not in same hemisphere (matches pbrt-v4 line 313-314)
        if !same_hemisphere(wo, wi)
            return SpectralBSDFSample()
        end

        # Compute PDF of wi for microfacet reflection (matches pbrt-v4 line 317)
        # pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm))
        pdf = trowbridge_reitz_pdf(wo, wm, alpha_x, alpha_y) / (4f0 * abs(dot(wo, wm)))

        # Get cos values (matches pbrt-v4 line 319-321)
        cos_theta_o = abs_cos_theta(wo)
        cos_theta_i = abs_cos_theta(wi)
        if cos_theta_i == 0f0 || cos_theta_o == 0f0
            return SpectralBSDFSample()
        end

        # Evaluate Fresnel factor F for conductor BRDF (matches pbrt-v4 line 323)
        # FrComplex uses AbsDot(wo, wm), not AbsCosTheta
        F = fr_complex_spectral(abs(dot(wo, wm)), eta_spectral, k_spectral)

        # Compute BSDF value (matches pbrt-v4 line 325-326)
        # f = D(wm) * F * G(wo, wi) / (4 * cosTheta_i * cosTheta_o)
        D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
        G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
        f = D * F * G / (4f0 * cos_theta_i * cos_theta_o)

        # Transform wi back to world coordinates
        wi_world = local_to_world(wi, n, tangent, bitangent)

        return SpectralBSDFSample(f, wi_world, pdf, BXDF_GLOSSY_REFLECTION, 1f0)
    end
end

# ============================================================================
# Conductor spectral BSDF evaluation (for MIS in direct lighting)
# ============================================================================

"""
    evaluate_bsdf_spectral(table, mat::Conductor, ...) -> (f, pdf)

Evaluate metal BSDF for given directions.
Matches pbrt-v4's ConductorBxDF::f and ConductorBxDF::PDF exactly.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    mat::Conductor, table::RGBToSpectrumTable, textures,
    wo_world::Vec3f, wi_world::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths,
    regularize::Bool = false
)
    # Build local coordinate frame
    tangent, bitangent = shading_frame(n, dpdus)

    # Transform to local coordinates
    wo = world_to_local(wo_world, n, tangent, bitangent)
    wi = world_to_local(wi_world, n, tangent, bitangent)

    # Must be in same hemisphere (matches pbrt-v4 line 332-333)
    if !same_hemisphere(wo, wi)
        return (SpectralRadiance(), 0f0)
    end

    # Get material properties
    roughness = eval_tex(textures, mat.roughness, tfc)

    # Compute alpha values
    alpha_x = mat.remap_roughness ? roughness_to_α(roughness) : roughness
    alpha_y = alpha_x

    # Apply regularization to match sampling state
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end

    # Clamp alpha if not smooth
    if !trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        alpha_x = max(alpha_x, 1f-4)
        alpha_y = max(alpha_y, 1f-4)
    end

    # Specular returns zero for evaluation (matches pbrt-v4 line 334-335)
    if trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        return (SpectralRadiance(), 0f0)
    end

    # Evaluate rough conductor BRDF (matches pbrt-v4 line 336-350)
    cos_theta_o = abs_cos_theta(wo)
    cos_theta_i = abs_cos_theta(wi)
    if cos_theta_i == 0f0 || cos_theta_o == 0f0
        return (SpectralRadiance(), 0f0)
    end

    # Compute half-vector wm (matches pbrt-v4 line 341-344)
    wm = wi + wo
    if dot(wm, wm) == 0f0
        return (SpectralRadiance(), 0f0)
    end
    wm = normalize(wm)

    # Evaluate eta and k spectrally (dispatches on PiecewiseLinearSpectrum vs RGB texture)
    eta_spectral = eval_ior_spectral(table, textures, mat.eta, tfc, lambda)
    k_spectral = eval_ior_spectral(table, textures, mat.k, tfc, lambda)

    # Evaluate Fresnel factor F (matches pbrt-v4 line 347)
    F = fr_complex_spectral(abs(dot(wo, wm)), eta_spectral, k_spectral)

    # Compute BSDF value (matches pbrt-v4 line 349)
    # f = D(wm) * F * G(wo, wi) / (4 * cosTheta_i * cosTheta_o)
    D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
    G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
    f = D * F * G / (4f0 * cos_theta_i * cos_theta_o)

    # Compute PDF (matches pbrt-v4 line 361-367)
    # wm needs to face forward for PDF
    wm_pdf = face_forward(wm, Vec3f(0f0, 0f0, 1f0))
    pdf = trowbridge_reitz_pdf(wo, wm_pdf, alpha_x, alpha_y) / (4f0 * abs(dot(wo, wm_pdf)))

    return (f, pdf)
end
