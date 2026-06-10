# ============================================================================
# DiffuseTransmission - Diffuse reflection and transmission
# ============================================================================
# Port of pbrt-v4's DiffuseTransmission and DiffuseTransmissionBxDF
#
# This material models surfaces that scatter light diffusely in both
# reflection and transmission, like thin cloth, paper, or leaves.
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h DiffuseTransmissionBxDF (lines 84-164)

"""
    DiffuseTransmission{RTex, TTex}

A material that diffusely reflects and transmits light.

Models surfaces like paper, thin fabric, or leaves where light scatters
diffusely on both sides. The reflection and transmission are independent
Lambertian distributions.

# Fields
- `reflectance`: Diffuse reflectance color (same hemisphere as incident)
- `transmittance`: Diffuse transmittance color (opposite hemisphere)
- `scale`: Intensity multiplier applied to both R and T

# Physics
- Reflection: f = R/π (same hemisphere)
- Transmission: f = T/π (opposite hemisphere)
- Sampling: probability proportional to max(R) and max(T)

# Usage
```julia
# Thin white paper (equal reflection and transmission)
paper = DiffuseTransmission(reflectance=(0.8, 0.8, 0.8), transmittance=(0.5, 0.5, 0.5))

# Green leaf (green transmission, less reflection)
leaf = DiffuseTransmission(reflectance=(0.2, 0.3, 0.1), transmittance=(0.1, 0.5, 0.1))
```
"""
struct DiffuseTransmission{RTex, TTex} <: Material
    reflectance::RTex    # Texture{RGBSpectrum} - diffuse reflection
    transmittance::TTex  # Texture{RGBSpectrum} - diffuse transmission
    scale::Float32       # Intensity scale
end

# Full constructor. Texture args are unannotated on purpose: fields accept any
# texture-like value (Texture, CheckerboardTexture, raw constants, TextureRef).
function DiffuseTransmission(
    reflectance,
    transmittance,
    scale::Float32
)
    DiffuseTransmission{typeof(reflectance), typeof(transmittance)}(
        reflectance, transmittance, scale
    )
end

"""
    DiffuseTransmission(; reflectance, transmittance, scale=1.0)

Create a diffuse transmission material with keyword arguments.

# Arguments
- `reflectance`: Diffuse reflection color (RGBSpectrum, tuple, or Texture)
- `transmittance`: Diffuse transmission color (RGBSpectrum, tuple, or Texture)
- `scale`: Intensity multiplier (default 1.0)

# Examples
```julia
# Thin translucent material
DiffuseTransmission(reflectance=(0.5, 0.5, 0.5), transmittance=(0.3, 0.3, 0.3))

# Pure transmission (no reflection)
DiffuseTransmission(reflectance=(0, 0, 0), transmittance=(1, 1, 1))
```
"""
function DiffuseTransmission(;
    reflectance = RGBSpectrum(0.5f0),
    transmittance = RGBSpectrum(0.5f0),
    scale::Real = 1f0
)
    DiffuseTransmission(
        to_texture(reflectance),
        to_texture(transmittance),
        Float32(scale)
    )
end

# Mark as non-emissive
is_emissive(::DiffuseTransmission) = false


# ============================================================================
# Spectral BSDF Evaluation — DiffuseTransmission
# ============================================================================

"""
    sample_bsdf_spectral(mat::DiffuseTransmission, ...) -> SpectralBSDFSample

Sample diffuse transmission BSDF matching pbrt-v4's DiffuseTransmissionBxDF::Sample_f.

This material diffusely scatters light in both reflection (same hemisphere)
and transmission (opposite hemisphere). Sampling is proportional to max(R) and max(T).
"""
@propagate_inbounds function sample_bsdf_spectral(
    mat::DiffuseTransmission, table::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties and apply scale
    r_rgb = eval_tex(textures, mat.reflectance, tfc) * mat.scale
    t_rgb = eval_tex(textures, mat.transmittance, tfc) * mat.scale

    # Clamp to [0, 1]
    r_rgb = RGBSpectrum(clamp(r_rgb.c[1], 0f0, 1f0), clamp(r_rgb.c[2], 0f0, 1f0), clamp(r_rgb.c[3], 0f0, 1f0))
    t_rgb = RGBSpectrum(clamp(t_rgb.c[1], 0f0, 1f0), clamp(t_rgb.c[2], 0f0, 1f0), clamp(t_rgb.c[3], 0f0, 1f0))

    r_spectral = uplift_rgb(table, r_rgb, lambda)
    t_spectral = uplift_rgb(table, t_rgb, lambda)

    # Compute probabilities based on max component (pbrt-v4 lines 102-108)
    pr = max(r_rgb.c[1], r_rgb.c[2], r_rgb.c[3])
    pt = max(t_rgb.c[1], t_rgb.c[2], t_rgb.c[3])

    if pr + pt < 1f-10
        return SpectralBSDFSample()
    end

    # Build local coordinate frame
    tangent, bitangent = shading_frame(n, dpdus)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    prob_reflect = pr / (pr + pt)

    if rng < prob_reflect
        # Sample diffuse reflection (same hemisphere as wo)
        local_wi = cosine_sample_hemisphere(sample_u)

        # Flip to same hemisphere as wo
        if wo_local[3] < 0f0
            local_wi = Vec3f(local_wi[1], local_wi[2], -local_wi[3])
        end

        cos_theta = abs(local_wi[3])
        if cos_theta < 1f-6
            return SpectralBSDFSample()
        end

        wi = tangent * local_wi[1] + bitangent * local_wi[2] + n * local_wi[3]
        wi = normalize(wi)

        # f = R / π
        f_spectral = r_spectral * (1f0 / Float32(π))
        pdf = prob_reflect * cos_theta / Float32(π)

        return SpectralBSDFSample(f_spectral, wi, pdf, BXDF_DIFFUSE_REFLECTION, 1f0)
    else
        # Sample diffuse transmission (opposite hemisphere from wo)
        local_wi = cosine_sample_hemisphere(sample_u)

        # Flip to opposite hemisphere from wo
        if wo_local[3] > 0f0
            local_wi = Vec3f(local_wi[1], local_wi[2], -local_wi[3])
        end

        cos_theta = abs(local_wi[3])
        if cos_theta < 1f-6
            return SpectralBSDFSample()
        end

        wi = tangent * local_wi[1] + bitangent * local_wi[2] + n * local_wi[3]
        wi = normalize(wi)

        # f = T / π
        f_spectral = t_spectral * (1f0 / Float32(π))
        pdf = (1f0 - prob_reflect) * cos_theta / Float32(π)

        return SpectralBSDFSample(f_spectral, wi, pdf, BXDF_DIFFUSE_TRANSMISSION, 1f0)
    end
end

"""
    evaluate_bsdf_spectral(table, mat::DiffuseTransmission, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate diffuse transmission BSDF matching pbrt-v4's DiffuseTransmissionBxDF::f and PDF.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    mat::DiffuseTransmission, table::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths,
    regularize::Bool = false
)
    cos_θi = dot(wi, n)
    cos_θo = dot(wo, n)

    abs_cos_θi = abs(cos_θi)
    if abs_cos_θi < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Get material properties and apply scale
    r_rgb = eval_tex(textures, mat.reflectance, tfc) * mat.scale
    t_rgb = eval_tex(textures, mat.transmittance, tfc) * mat.scale

    # Clamp to [0, 1]
    r_rgb = RGBSpectrum(clamp(r_rgb.c[1], 0f0, 1f0), clamp(r_rgb.c[2], 0f0, 1f0), clamp(r_rgb.c[3], 0f0, 1f0))
    t_rgb = RGBSpectrum(clamp(t_rgb.c[1], 0f0, 1f0), clamp(t_rgb.c[2], 0f0, 1f0), clamp(t_rgb.c[3], 0f0, 1f0))

    r_spectral = uplift_rgb(table, r_rgb, lambda)
    t_spectral = uplift_rgb(table, t_rgb, lambda)

    # Probabilities
    pr = max(r_rgb.c[1], r_rgb.c[2], r_rgb.c[3])
    pt = max(t_rgb.c[1], t_rgb.c[2], t_rgb.c[3])

    if pr + pt < 1f-10
        return (SpectralRadiance(), 0f0)
    end

    same_hemisphere = (cos_θi * cos_θo) > 0f0

    if same_hemisphere
        # Reflection: f = R / π
        f_spectral = r_spectral * (1f0 / Float32(π))
        prob_reflect = pr / (pr + pt)
        pdf = prob_reflect * abs_cos_θi / Float32(π)
        return (f_spectral, pdf)
    else
        # Transmission: f = T / π
        f_spectral = t_spectral * (1f0 / Float32(π))
        prob_transmit = pt / (pr + pt)
        pdf = prob_transmit * abs_cos_θi / Float32(π)
        return (f_spectral, pdf)
    end
end
