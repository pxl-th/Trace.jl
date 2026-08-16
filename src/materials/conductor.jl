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

# MirrorEvaluated — Kr resolved once per hit (see `get_bxdf`).
struct MirrorEvaluated
    kr::SpectralRadiance   # already uplifted to the 4 wavelengths
end

@propagate_inbounds get_bxdf(
    mat::Mirror, table::RGBToSpectrumTable, textures,
    tfc::TextureFilterContext, lambda::Wavelengths, ::Bool,
) = MirrorEvaluated(uplift_rgb(table, eval_tex(textures, mat.Kr, tfc), lambda))

"""
    sample_bsdf_spectral(bxdf::MirrorEvaluated, table, textures, wo, n, dpdus, tfc, lambda, sample_u, rng) -> SpectralBSDFSample

Sample perfect specular reflection with spectral evaluation.
"""
@propagate_inbounds function sample_bsdf_spectral(
    bxdf::MirrorEvaluated, ::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext,
    ::Wavelengths, sample_u::Point2f, ::Float32,
    ::Bool = false,
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Reflectance was resolved once in `get_bxdf` (rng and sample_u are unused
    # for perfect specular).
    kr_spectral = bxdf.kr

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
    ::MirrorEvaluated, ::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext, ::Wavelengths,
    ::Bool = false,
)
    # Perfect specular has zero PDF for non-delta directions
    return (SpectralRadiance(), 0f0)
end


# ============================================================================
# ConductorEvaluated — pbrt-v4 ConductorBxDF analogue
# ============================================================================
#
# pbrt-v4's wavefront `EvaluateMaterialAndBSDF<ConductorMaterial>` calls
# `Material::GetBxDF(texEval, ctx, lambda)` ONCE per surface hit; that's where
# `mat.eta` / `mat.k` (SpectrumTexture handles, e.g. `PiecewiseLinearSpectrum`)
# get evaluated at the 4 sampled wavelengths into the 4-float `SampledSpectrum
# eta, k` that `ConductorBxDF` carries.  After that the BxDF is a 40-byte
# struct {mfDistrib, eta, k}; `Sample_f`, `f`, `PDF` all operate on the
# already-resolved spectra — no further binary search through the 56-element
# table per call.
#
# Hikari's `Conductor{PiecewiseLinearSpectrum{56}, …}` is ~900 bytes and the
# previous `sample_bsdf_spectral(::Conductor)` / `evaluate_bsdf_spectral
# (::Conductor)` methods each re-did `eval_ior_spectral(mat.eta, lambda)` and
# `eval_ior_spectral(mat.k, lambda)` — i.e. 8 binary searches per call, and
# both BSDF calls happen on every surface hit (sample + MIS eval), giving
# 16-24 binary searches per surface hit AND inflating live state with the
# full 900-byte Conductor struct.
#
# `ConductorEvaluated` matches pbrt-v4's `ConductorBxDF`: alpha_x/y resolved
# (and pre-regularized + pre-clamped), eta/k resolved to a 4-element
# SpectralRadiance.  `get_bxdf(::Conductor, ...)` constructs it once per hit;
# subsequent `sample_bsdf_spectral(::ConductorEvaluated, ...)` and
# `evaluate_bsdf_spectral(::ConductorEvaluated, ...)` calls only see this
# small struct, so the kernel's register pressure on the Conductor path drops
# closer to the Diffuse baseline.
struct ConductorEvaluated
    alpha_x::Float32
    alpha_y::Float32
    eta::SpectralRadiance       # already sampled at the 4 wavelengths
    k::SpectralRadiance         # already sampled at the 4 wavelengths
end

"""
    get_bxdf(mat, table, textures, tfc, lambda, regularize) -> bxdf

Hikari's analogue of pbrt-v4's `Material::GetBxDF`. Resolves any per-hit
spectral / textured material state into a small per-hit BSDF carrier whose
methods don't repeat that work. Default falls back to identity — most
materials are already small (Diffuse is RGB+Float = 16 bytes), so the
overhead of an intermediate struct is unjustified for them.
"""
@inline get_bxdf(mat, table, textures, tfc, lambda, regularize::Bool) = mat

@propagate_inbounds function get_bxdf(
    mat::Conductor, table::RGBToSpectrumTable, textures,
    tfc::TextureFilterContext, lambda::Wavelengths, regularize::Bool,
)
    roughness = eval_tex(textures, mat.roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(roughness) : roughness
    alpha_y = alpha_x  # isotropic
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end
    if !trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        alpha_x = max(alpha_x, 1f-4)
        alpha_y = max(alpha_y, 1f-4)
    end
    eta = eval_ior_spectral(table, textures, mat.eta, tfc, lambda)
    k   = eval_ior_spectral(table, textures, mat.k,   tfc, lambda)
    return ConductorEvaluated(alpha_x, alpha_y, eta, k)
end

# ----------------------------------------------------------------------------
# sample_bsdf_spectral(::ConductorEvaluated, …) — pbrt-v4 ConductorBxDF::Sample_f
# ----------------------------------------------------------------------------
# Mirrors `bxdfs.h` ConductorBxDF::Sample_f exactly (lines 296-328).
# Ignores the trailing `tfc` / `regularize` — both were folded into the
# pre-built bxdf at `get_bxdf` time.
function sample_bsdf_spectral(
    bxdf::ConductorEvaluated, ::RGBToSpectrumTable, textures,
    wo_world::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext,
    ::Wavelengths, sample_u::Point2f, ::Float32,
    ::Bool = false,
)
    tangent, bitangent = shading_frame(n, dpdus)
    wo = world_to_local(wo_world, n, tangent, bitangent)
    if wo[3] == 0f0
        return SpectralBSDFSample()
    end

    alpha_x = bxdf.alpha_x
    alpha_y = bxdf.alpha_y

    if trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        wi = Vec3f(-wo[1], -wo[2], wo[3])
        cos_theta_i = abs_cos_theta(wi)
        F = fr_complex_spectral(cos_theta_i, bxdf.eta, bxdf.k)
        f = F / cos_theta_i
        wi_world = local_to_world(wi, n, tangent, bitangent)
        return SpectralBSDFSample(f, wi_world, 1f0, BXDF_SPECULAR_REFLECTION, 1f0)
    end

    wm = trowbridge_reitz_sample_wm(wo, sample_u, alpha_x, alpha_y)
    wi = -wo + 2f0 * dot(wo, wm) * wm
    if !same_hemisphere(wo, wi)
        return SpectralBSDFSample()
    end
    pdf = trowbridge_reitz_pdf(wo, wm, alpha_x, alpha_y) / (4f0 * abs(dot(wo, wm)))
    cos_theta_o = abs_cos_theta(wo)
    cos_theta_i = abs_cos_theta(wi)
    if cos_theta_i == 0f0 || cos_theta_o == 0f0
        return SpectralBSDFSample()
    end
    F = fr_complex_spectral(abs(dot(wo, wm)), bxdf.eta, bxdf.k)
    D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
    G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
    f = D * F * G / (4f0 * cos_theta_i * cos_theta_o)
    wi_world = local_to_world(wi, n, tangent, bitangent)
    return SpectralBSDFSample(f, wi_world, pdf, BXDF_GLOSSY_REFLECTION, 1f0)
end

# ----------------------------------------------------------------------------
# evaluate_bsdf_spectral(::ConductorEvaluated, …) — pbrt-v4 ConductorBxDF::f / PDF
# ----------------------------------------------------------------------------
function evaluate_bsdf_spectral(
    bxdf::ConductorEvaluated, ::RGBToSpectrumTable, textures,
    wo_world::Vec3f, wi_world::Vec3f, n::Vec3f, dpdus::Vec3f,
    ::TextureFilterContext, ::Wavelengths,
    ::Bool = false,
)
    tangent, bitangent = shading_frame(n, dpdus)
    wo = world_to_local(wo_world, n, tangent, bitangent)
    wi = world_to_local(wi_world, n, tangent, bitangent)
    if !same_hemisphere(wo, wi)
        return (SpectralRadiance(), 0f0)
    end

    alpha_x = bxdf.alpha_x
    alpha_y = bxdf.alpha_y

    # Perfectly smooth → delta distribution; non-delta directions have zero
    # density (matches pbrt-v4 ConductorBxDF::f line 334-335).
    if trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        return (SpectralRadiance(), 0f0)
    end

    cos_theta_o = abs_cos_theta(wo)
    cos_theta_i = abs_cos_theta(wi)
    if cos_theta_i == 0f0 || cos_theta_o == 0f0
        return (SpectralRadiance(), 0f0)
    end
    wm = wi + wo
    if dot(wm, wm) == 0f0
        return (SpectralRadiance(), 0f0)
    end
    wm = normalize(wm)
    F = fr_complex_spectral(abs(dot(wo, wm)), bxdf.eta, bxdf.k)
    D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
    G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
    f = D * F * G / (4f0 * cos_theta_i * cos_theta_o)
    wm_pdf = face_forward(wm, Vec3f(0f0, 0f0, 1f0))
    pdf = trowbridge_reitz_pdf(wo, wm_pdf, alpha_x, alpha_y) / (4f0 * abs(dot(wo, wm_pdf)))
    return (f, pdf)
end
