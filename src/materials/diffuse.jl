# ============================================================================
# Diffuse (Matte) Material
# Contains: Diffuse struct, keyword constructor, Diffuse alias,
#           sample_bsdf_spectral and evaluate_bsdf_spectral methods
# ============================================================================

"""
    Diffuse(Kd::Texture, σ::Texture)

Matte (diffuse) material with Lambertian or Oren-Nayar BRDF.

* `Kd`: Spectral diffuse reflection (color texture or TextureRef)
* `σ`: Scalar roughness for Oren-Nayar model (0 = Lambertian)
"""
struct Diffuse{KdTex, σTex} <: Material
    Kd::KdTex   # Texture, Raycore.TextureRef, or raw RGBSpectrum
    σ::σTex     # Texture, Raycore.TextureRef, or raw Float32
end


"""
    Diffuse(; Kd=RGBSpectrum(0.5), σ=0.0)

Create a matte (diffuse) material with optional Oren-Nayar roughness.

# Arguments
- `Kd`: Diffuse color - can be RGBSpectrum, (r,g,b) tuple, or Texture
- `σ`: Roughness angle in degrees (0 = Lambertian, >0 = Oren-Nayar)

# Examples
```julia
Diffuse(Kd=RGBSpectrum(0.8, 0.2, 0.2))  # Red matte
Diffuse(Kd=(0.8, 0.2, 0.2), σ=20)       # Red with roughness
Diffuse(Kd=my_texture)                   # Textured
```
"""
function Diffuse(; Kd=RGBSpectrum(0.5f0), σ=0f0)
    Diffuse(to_texture(Kd), to_texture(σ))
end


# ============================================================================
# Spectral BSDF sampling
# ============================================================================

"""
    sample_bsdf_spectral(table, mat::Diffuse, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample diffuse BSDF with spectral evaluation.
Uses pbrt-v4 convention: work in local shading space where n = (0,0,1).
"""
@propagate_inbounds function sample_bsdf_spectral(
    mat::Diffuse, table::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle (wo perpendicular to shading normal)
    # This matches pbrt-v4's wo.z == 0 check in BSDF::Sample_f
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties
    # Alpha is handled at intersection level (vp_trace_rays_kernel!), not here.
    kd_rgb = eval_tex(textures, mat.Kd, tfc)
    σ = eval_tex(textures, mat.σ, tfc)

    # Clamp reflectance to [0,1] as per pbrt-v4
    kd_rgb = clamp(kd_rgb)

    # Uplift to spectral
    kd_spectral = uplift_rgb(table, kd_rgb, lambda)

    # Build local coordinate system from shading normal
    tangent, bitangent = shading_frame(n, dpdus)

    # Cosine-weighted hemisphere sampling (in local space, normal = +z)
    local_wi = cosine_sample_hemisphere(sample_u)
    cos_theta = local_wi[3]

    if cos_theta < 1f-6
        return SpectralBSDFSample()
    end

    # If wo is on the backside of the shading normal, flip wi to same hemisphere
    # This matches pbrt-v4's: if (wo.z < 0) wi.z *= -1;
    if wo_dot_n < 0f0
        local_wi = Vec3f(local_wi[1], local_wi[2], -local_wi[3])
    end

    # Transform to world space
    wi = local_to_world(local_wi, n, tangent, bitangent)
    wi = normalize(wi)

    # f = Kd / π (Lambertian), pdf = cos_theta / π
    # For Oren-Nayar with σ > 0, use the full model
    if σ > 0f0
        # Simplified Oren-Nayar: f ≈ Kd/π * (A + B * max(0, cos(φi-φo)) * sin(α) * tan(β))
        # For simplicity, we use a roughness-scaled Lambertian approximation
        roughness_factor = 1f0 - 0.5f0 * σ / (σ + 0.33f0)
        f = kd_spectral * (roughness_factor / Float32(π))
    else
        f = kd_spectral * (1f0 / Float32(π))
    end

    pdf = cos_theta / Float32(π)

    return SpectralBSDFSample(f, wi, pdf, BXDF_DIFFUSE_REFLECTION, 1f0)
end

# ============================================================================
# Spectral BSDF evaluation (for MIS in direct lighting)
# ============================================================================

@propagate_inbounds function evaluate_bsdf_spectral(
    mat::Diffuse, table::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths,
    regularize::Bool = false
)
    # Check if wi is in the correct hemisphere
    cos_theta_i = dot(wi, n)
    cos_theta_o = dot(wo, n)
    if cos_theta_i * cos_theta_o < 0f0
        return (SpectralRadiance(), 0f0)
    end

    cos_theta = abs(cos_theta_i)
    if cos_theta < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Alpha is handled at intersection level (vp_trace_rays_kernel!), not here.
    kd_rgb = eval_tex(textures, mat.Kd, tfc)

    # Clamp reflectance to [0,1] as per pbrt-v4
    kd_rgb = clamp(kd_rgb)
    kd_spectral = uplift_rgb(table, kd_rgb, lambda)
    f = kd_spectral / Float32(π)
    pdf = cos_theta / Float32(π)

    return (f, pdf)
end

# ============================================================================
# Surface Alpha Evaluation
# ============================================================================

@propagate_inbounds function get_surface_alpha(mat::Diffuse, textures, uv::Point2f)
    kd_rgb = eval_tex(textures, mat.Kd, uv)
    return get_alpha(kd_rgb)
end
