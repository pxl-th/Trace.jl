# ============================================================================
# CoatedConductor - Layered material with dielectric coating over conductor
# ============================================================================
# Port of pbrt-v4's CoatedConductor using LayeredBxDF
#
# The material consists of:
# - Top layer: Dielectric interface (can be rough or smooth)
# - Bottom layer: Conductor (metal)
# - Optional absorbing medium between layers
#
# Reference: pbrt-v4 src/pbrt/materials.cpp CoatedConductor::GetBxDF (lines 345-392)
# Reference: pbrt-v4 src/pbrt/bxdfs.h CoatedConductorBxDF (lines 911-918)

"""
    CoatedConductor

A layered material with a dielectric coating over a conductor (metal) base.
This implements pbrt-v4's coatedconductor material using random walk
sampling between the layers (LayeredBxDF algorithm).

# Fields
## Interface (coating) layer
- `interface_u_roughness`: U roughness for the dielectric coating
- `interface_v_roughness`: V roughness for the dielectric coating
- `interface_eta`: Index of refraction of the dielectric coating

## Conductor (base) layer
- `conductor_eta`: Complex index of refraction (real part) - OR use reflectance
- `conductor_k`: Complex index of refraction (imaginary part)
- `reflectance`: Alternative to eta/k - artist-friendly reflectance color
- `conductor_u_roughness`: U roughness for the conductor
- `conductor_v_roughness`: V roughness for the conductor

## Volumetric scattering (between layers)
- `thickness`: Thickness of the coating layer (affects absorption)
- `albedo`: Single-scattering albedo of medium between layers (0 = no absorption)
- `g`: Henyey-Greenstein asymmetry parameter for medium scattering

## Random walk parameters
- `max_depth`: Maximum random walk depth
- `n_samples`: Number of samples for estimating the BSDF
- `remap_roughness`: Whether to remap roughness to microfacet alpha

# Notes
- **Critical:** Conductor eta/k are scaled by interface IOR: ce /= ieta, ck /= ieta
- If `conductor_eta` is nothing, uses reflectance-based approach
"""
struct CoatedConductor{
    IURoughTex, IVRoughTex,
    CETex, CKTex, CReflTex, CURoughTex, CVRoughTex,
    ThickTex, AlbedoTex, GTex
} <: Material
    # Interface parameters
    interface_u_roughness::IURoughTex  # Texture{Float32}
    interface_v_roughness::IVRoughTex  # Texture{Float32}
    interface_eta::Float32             # Scalar IOR for interface

    # Conductor parameters (either eta/k OR reflectance)
    conductor_eta::CETex               # Texture{RGBSpectrum} - complex IOR real part (or nothing)
    conductor_k::CKTex                 # Texture{RGBSpectrum} - complex IOR imaginary part
    reflectance::CReflTex              # Texture{RGBSpectrum} - alternative to eta/k
    conductor_u_roughness::CURoughTex  # Texture{Float32}
    conductor_v_roughness::CVRoughTex  # Texture{Float32}

    # Volumetric scattering
    thickness::ThickTex                # Texture{Float32}
    albedo::AlbedoTex                  # Texture{RGBSpectrum} - medium albedo
    g::GTex                            # Texture{Float32} - HG asymmetry

    # Algorithm parameters
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool

    # Mode flag: true if using eta/k, false if using reflectance
    use_eta_k::Bool
end

# Full constructor with all textures
function CoatedConductor(
    interface_u_roughness::Texture,
    interface_v_roughness::Texture,
    interface_eta::Float32,
    conductor_eta::Union{Texture, Nothing},
    conductor_k::Union{Texture, Nothing},
    reflectance::Union{Texture, Nothing},
    conductor_u_roughness::Texture,
    conductor_v_roughness::Texture,
    thickness::Texture,
    albedo::Texture,
    g::Texture,
    max_depth::Int,
    n_samples::Int,
    remap_roughness::Bool
)
    use_eta_k = !isnothing(conductor_eta)

    # If not using eta/k, set them to dummy values (raw values for constants)
    ce = isnothing(conductor_eta) ? RGBSpectrum(1f0) : conductor_eta
    ck = isnothing(conductor_k) ? RGBSpectrum(0f0) : conductor_k
    refl = isnothing(reflectance) ? RGBSpectrum(1f0) : reflectance

    CoatedConductor{
        typeof(interface_u_roughness), typeof(interface_v_roughness),
        typeof(ce), typeof(ck), typeof(refl),
        typeof(conductor_u_roughness), typeof(conductor_v_roughness),
        typeof(thickness), typeof(albedo), typeof(g)
    }(
        interface_u_roughness, interface_v_roughness, interface_eta,
        ce, ck, refl,
        conductor_u_roughness, conductor_v_roughness,
        thickness, albedo, g,
        Int32(max_depth), Int32(n_samples), remap_roughness,
        use_eta_k
    )
end

"""
    CoatedConductor(; interface_roughness=0.0, interface_eta=1.5, ...)

Create a coated conductor material with keyword arguments.

# Arguments
## Interface (coating)
- `interface_roughness`: Coating roughness (scalar or (u,v) tuple), default 0
- `interface_eta`: Coating IOR (default 1.5)

## Conductor (base) - use EITHER eta/k OR reflectance
- `conductor_eta`: Complex IOR real part (RGBSpectrum, tuple, or Texture)
- `conductor_k`: Complex IOR imaginary part
- `reflectance`: Alternative artist-friendly color (if eta/k not specified)
- `conductor_roughness`: Conductor roughness (scalar or (u,v) tuple), default 0.01

## Volumetric
- `thickness`: Coating thickness (default 0.01)
- `albedo`: Medium albedo (default 0 = no medium)
- `g`: HG asymmetry (default 0 = isotropic)

## Algorithm
- `max_depth`: Max random walk depth (default 10)
- `n_samples`: Number of samples (default 1)
- `remap_roughness`: Remap roughness to alpha (default true)

# Examples
```julia
# Glossy coated gold
CoatedConductor(
    interface_roughness=0.05,
    conductor_eta=(0.143, 0.374, 1.442),  # Gold
    conductor_k=(3.983, 2.385, 1.603)
)

# Coated copper using reflectance
CoatedConductor(
    reflectance=(0.95, 0.64, 0.54),  # Copper-like
    conductor_roughness=0.1
)

# Car paint effect (rough coating over smooth metal)
CoatedConductor(
    interface_roughness=0.3,
    conductor_roughness=0.01,
    reflectance=(0.9, 0.1, 0.1)  # Red metallic
)
```
"""
function CoatedConductor(;
    # Interface parameters
    interface_roughness = 0f0,
    interface_eta::Real = 1.5f0,
    # Conductor parameters - eta/k mode
    conductor_eta = nothing,
    conductor_k = nothing,
    # Conductor parameters - reflectance mode
    reflectance = nothing,
    conductor_roughness = 0.01f0,
    # Volumetric
    thickness = 0.01f0,
    albedo = RGBSpectrum(0f0),
    g = 0f0,
    # Algorithm
    max_depth::Int = 10,
    n_samples::Int = 1,
    remap_roughness::Bool = true
)
    # Handle interface roughness - can be scalar or (u,v) tuple
    iu_rough, iv_rough = if interface_roughness isa Tuple
        Float32(interface_roughness[1]), Float32(interface_roughness[2])
    else
        Float32(interface_roughness), Float32(interface_roughness)
    end

    # Handle conductor roughness - can be scalar or (u,v) tuple
    cu_rough, cv_rough = if conductor_roughness isa Tuple
        Float32(conductor_roughness[1]), Float32(conductor_roughness[2])
    else
        Float32(conductor_roughness), Float32(conductor_roughness)
    end

    # Determine mode: eta/k or reflectance
    # If conductor_eta is provided, use eta/k mode; otherwise use reflectance mode
    if !isnothing(conductor_eta)
        # eta/k mode
        if isnothing(conductor_k)
            error("conductor_k must be provided when using conductor_eta")
        end
        CoatedConductor(
            to_texture(iu_rough),
            to_texture(iv_rough),
            Float32(interface_eta),
            to_texture(conductor_eta),
            to_texture(conductor_k),
            nothing,  # reflectance not used
            to_texture(cu_rough),
            to_texture(cv_rough),
            to_texture(Float32(thickness)),
            to_texture(albedo),
            to_texture(Float32(g)),
            max_depth,
            n_samples,
            remap_roughness
        )
    else
        # reflectance mode
        refl = isnothing(reflectance) ? RGBSpectrum(1f0) : reflectance
        CoatedConductor(
            to_texture(iu_rough),
            to_texture(iv_rough),
            Float32(interface_eta),
            nothing,  # eta not used
            nothing,  # k not used
            to_texture(refl),
            to_texture(cu_rough),
            to_texture(cv_rough),
            to_texture(Float32(thickness)),
            to_texture(albedo),
            to_texture(Float32(g)),
            max_depth,
            n_samples,
            remap_roughness
        )
    end
end

# Mark as non-emissive
is_emissive(::CoatedConductor) = false


# ============================================================================
# Spectral BSDF Evaluation — CoatedConductor
# ============================================================================

"""
    sample_bsdf_spectral(mat::CoatedConductor, ...) -> SpectralBSDFSample

Sample CoatedConductor BSDF using pbrt-v4's LayeredBxDF approach.

This is a layered material with:
- Top layer: Dielectric coating (can be rough or smooth)
- Bottom layer: Conductor (metal) with complex Fresnel

Key pbrt-v4 details (materials.cpp lines 345-392):
- Conductor eta/k are scaled by interface IOR: ce /= ieta, ck /= ieta
- If reflectance mode: k = 2 * sqrt(r) / sqrt(1 - r), eta = 1

When `regularize=true`, both interface and conductor microfacet alphas are increased
to reduce fireflies from near-specular paths (matches pbrt-v4 BSDF::Regularize).
"""
@propagate_inbounds function sample_bsdf_spectral(
    mat::CoatedConductor, table::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get interface (coating) parameters
    ieta = mat.interface_eta
    if ieta == 0f0
        ieta = 1f0
    end

    iu_roughness = eval_tex(textures, mat.interface_u_roughness, tfc)
    iv_roughness = eval_tex(textures, mat.interface_v_roughness, tfc)
    i_alpha_x = mat.remap_roughness ? roughness_to_α(iu_roughness) : iu_roughness
    i_alpha_y = mat.remap_roughness ? roughness_to_α(iv_roughness) : iv_roughness

    # Get conductor parameters
    cu_roughness = eval_tex(textures, mat.conductor_u_roughness, tfc)
    cv_roughness = eval_tex(textures, mat.conductor_v_roughness, tfc)
    c_alpha_x = mat.remap_roughness ? roughness_to_α(cu_roughness) : cu_roughness
    c_alpha_y = mat.remap_roughness ? roughness_to_α(cv_roughness) : cv_roughness

    # Apply regularization if requested (pbrt-v4: doubles alpha if < 0.3, clamps to [0.1, 0.3])
    if regularize
        i_alpha_x = regularize_alpha(i_alpha_x)
        i_alpha_y = regularize_alpha(i_alpha_y)
        c_alpha_x = regularize_alpha(c_alpha_x)
        c_alpha_y = regularize_alpha(c_alpha_y)
    end

    # Get conductor eta/k - either from eta/k textures or derived from reflectance
    local ce_spectral::SpectralRadiance
    local ck_spectral::SpectralRadiance

    if mat.use_eta_k
        ce_spectral = eval_ior_spectral(table, textures, mat.conductor_eta, tfc, lambda)
        ck_spectral = eval_ior_spectral(table, textures, mat.conductor_k, tfc, lambda)
    else
        # Reflectance mode: eta = 1, k = 2 * sqrt(r) / sqrt(1 - r)
        refl_rgb = eval_tex(textures, mat.reflectance, tfc)
        # Clamp to avoid r==1 NaN (pbrt-v4 line 371)
        refl_rgb = RGBSpectrum(
            clamp(refl_rgb.c[1], 0f0, 0.9999f0),
            clamp(refl_rgb.c[2], 0f0, 0.9999f0),
            clamp(refl_rgb.c[3], 0f0, 0.9999f0)
        )
        r_spectral = uplift_rgb(table, refl_rgb, lambda)
        ce_spectral = SpectralRadiance(1f0)
        # k = 2 * sqrt(r) / sqrt(1 - r) (pbrt-v4 line 373)
        ck_spectral = 2f0 * sqrt(r_spectral) / sqrt(clamp_zero(SpectralRadiance(1f0) - r_spectral) + SpectralRadiance(1f-6))
    end

    # Critical: scale conductor eta/k by interface IOR (pbrt-v4 lines 375-376)
    ce_spectral = ce_spectral / ieta
    ck_spectral = ck_spectral / ieta

    # Volumetric parameters
    thickness = max(eval_tex(textures, mat.thickness, tfc), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, tfc)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    g_val = clamp(eval_tex(textures, mat.g, tfc), -0.99f0, 0.99f0)
    has_medium = !is_black(albedo_rgb)

    # Build coordinate system from shading normal
    tangent, bitangent = coordinate_system(n)

    # Transform wo to local space
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    # Two-sided: flip if entering from below
    flip = wo_local[3] < 0f0
    if flip
        wo_local = -wo_local
    end

    cos_θo = abs(wo_local[3])

    # Check if interface coating is effectively smooth
    i_is_smooth = trowbridge_reitz_effectively_smooth(i_alpha_x, i_alpha_y)
    c_is_smooth = trowbridge_reitz_effectively_smooth(c_alpha_x, c_alpha_y)

    if i_is_smooth
        # === Smooth interface coating ===
        F_interface = fresnel_dielectric(cos_θo, ieta)

        if rng < F_interface
            # Specular reflection at coating surface
            wi_local = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])
            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            f_spectral = SpectralRadiance(1f0)
            return SpectralBSDFSample(wi, f_spectral, 1f0, true, 1f0)
        end

        # Transmitted through interface - now sample conductor base
        # Refract direction into coating
        sin2_θt = max(0f0, 1f0 - cos_θo^2) / (ieta^2)
        if sin2_θt >= 1f0
            return SpectralBSDFSample()  # TIR at interface
        end
        cos_θt_in = sqrt(1f0 - sin2_θt)

        if c_is_smooth
            # Smooth conductor: perfect reflection at base
            # wi in coating space points straight up after reflection
            wi_base = Vec3f(-wo_local[1] / ieta, -wo_local[2] / ieta, cos_θt_in)
            wi_base = normalize(wi_base)

            # Conductor Fresnel (using scaled eta/k)
            F_conductor = fr_complex_spectral(cos_θt_in, ce_spectral, ck_spectral)

            # Refract back out through interface
            sin2_θ_out = max(0f0, 1f0 - wi_base[3]^2) * (ieta^2)
            if sin2_θ_out >= 1f0
                return SpectralBSDFSample()  # TIR on way out
            end
            cos_θ_out = sqrt(1f0 - sin2_θ_out)

            F_out = fresnel_dielectric(cos_θ_out, ieta)
            T_in = 1f0 - F_interface
            T_out = 1f0 - F_out

            # Layer transmittance through medium
            layer_tr = if has_medium
                tr = layer_transmittance(thickness, Vec3f(0, 0, cos_θt_in))
                tr * tr * albedo_spectral
            else
                SpectralRadiance(1f0)
            end

            # Final direction is reflection
            wi_local = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])

            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            f_spectral = F_conductor * T_in * T_out * layer_tr / cos_θo
            return SpectralBSDFSample(wi, f_spectral, 1f0 - F_interface, true, 1f0)
        else
            # Rough conductor: sample microfacet
            # Transform wo to conductor local frame (inside coating)
            wo_conductor = Vec3f(wo_local[1] / ieta, wo_local[2] / ieta, cos_θt_in)
            wo_conductor = normalize(wo_conductor)

            # Clamp conductor alpha
            c_alpha_x = max(c_alpha_x, 1f-4)
            c_alpha_y = max(c_alpha_y, 1f-4)

            # Sample conductor microfacet
            wm = trowbridge_reitz_sample_wm(wo_conductor, sample_u, c_alpha_x, c_alpha_y)
            cos_θo_m = dot(wo_conductor, wm)
            if cos_θo_m < 0f0
                return SpectralBSDFSample()
            end

            # Reflect off conductor microfacet
            wi_conductor = -wo_conductor + 2f0 * cos_θo_m * wm
            if wi_conductor[3] < 0f0
                return SpectralBSDFSample()
            end

            # Conductor Fresnel at microfacet
            F_conductor = fr_complex_spectral(abs(cos_θo_m), ce_spectral, ck_spectral)

            # Conductor microfacet BRDF
            D = trowbridge_reitz_d(wm, c_alpha_x, c_alpha_y)
            G = trowbridge_reitz_g(wo_conductor, wi_conductor, c_alpha_x, c_alpha_y)
            f_conductor = D * F_conductor * G / (4f0 * abs(wo_conductor[3]) * abs(wi_conductor[3]))

            # Refract outgoing direction back through interface
            sin2_θ_out = (wi_conductor[1]^2 + wi_conductor[2]^2) * (ieta^2)
            if sin2_θ_out >= 1f0
                return SpectralBSDFSample()  # TIR
            end
            cos_θ_out = sqrt(1f0 - sin2_θ_out)

            F_out = fresnel_dielectric(cos_θ_out, ieta)
            T_in = 1f0 - F_interface
            T_out = 1f0 - F_out

            layer_tr = if has_medium
                tr_in = layer_transmittance(thickness, Vec3f(0, 0, cos_θt_in))
                tr_out = layer_transmittance(thickness, Vec3f(0, 0, wi_conductor[3]))
                tr_in * tr_out * albedo_spectral
            else
                SpectralRadiance(1f0)
            end

            # Transform wi back to world
            wi_local = Vec3f(wi_conductor[1] * ieta, wi_conductor[2] * ieta, cos_θ_out)
            wi_local = normalize(wi_local)

            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            f_spectral = f_conductor * T_in * T_out * layer_tr

            # PDF for conductor sampling
            pdf_m = trowbridge_reitz_pdf(wo_conductor, wm, c_alpha_x, c_alpha_y)
            pdf_conductor = pdf_m / (4f0 * abs(cos_θo_m))
            pdf = (1f0 - F_interface) * pdf_conductor

            return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
        end
    else
        # === Rough interface coating ===
        # Clamp interface alpha
        i_alpha_x = max(i_alpha_x, 1f-4)
        i_alpha_y = max(i_alpha_y, 1f-4)

        # Sample interface microfacet normal
        wm = trowbridge_reitz_sample_wm(wo_local, sample_u, i_alpha_x, i_alpha_y)
        cos_θo_m = dot(wo_local, wm)
        if cos_θo_m < 0f0
            return SpectralBSDFSample()
        end

        F_interface = fresnel_dielectric(cos_θo_m, ieta)

        if rng < F_interface
            # Reflect off interface microfacet
            wi_local = -wo_local + 2f0 * cos_θo_m * wm

            if wi_local[3] * wo_local[3] < 0f0
                return SpectralBSDFSample()
            end

            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            # Interface microfacet BRDF (dielectric)
            D = trowbridge_reitz_d(wm, i_alpha_x, i_alpha_y)
            G = trowbridge_reitz_g(wo_local, wi_local, i_alpha_x, i_alpha_y)

            cos_i = abs(wi_local[3])
            cos_o = abs(wo_local[3])

            pdf_m = trowbridge_reitz_pdf(wo_local, wm, i_alpha_x, i_alpha_y)
            pdf = F_interface * pdf_m / (4f0 * abs(cos_θo_m))

            f = D * G / (4f0 * cos_i * cos_o)

            return SpectralBSDFSample(wi, SpectralRadiance(f), pdf, false, 1f0)
        else
            # Transmit through rough interface to conductor
            # For rough interface, approximate with average transmission
            T_in = 1f0 - F_interface

            # Sample conductor - simplified for rough interface case
            # Use a cosine-weighted sample centered around specular direction
            local_conductor_wi = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])

            if c_is_smooth
                # Smooth conductor with rough interface
                cos_θ_base = abs(local_conductor_wi[3])
                F_conductor = fr_complex_spectral(cos_θ_base, ce_spectral, ck_spectral)

                F_out = fresnel_dielectric(cos_θ_base, ieta)
                T_out = 1f0 - F_out

                layer_tr = if has_medium
                    tr = layer_transmittance(thickness, local_conductor_wi)
                    tr * tr * albedo_spectral
                else
                    SpectralRadiance(1f0)
                end

                if flip
                    local_conductor_wi = -local_conductor_wi
                end

                wi = tangent * local_conductor_wi[1] + bitangent * local_conductor_wi[2] + n * local_conductor_wi[3]
                wi = normalize(wi)

                f_spectral = F_conductor * T_in * T_out * layer_tr / cos_θo

                pdf_m = trowbridge_reitz_pdf(wo_local, wm, i_alpha_x, i_alpha_y)
                pdf = (1f0 - F_interface) * pdf_m / (4f0 * abs(cos_θo_m))

                return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
            else
                # Both interface and conductor rough
                # Use the interface-sampled direction for conductor evaluation
                c_alpha_x = max(c_alpha_x, 1f-4)
                c_alpha_y = max(c_alpha_y, 1f-4)

                # Sample conductor from the transmitted direction
                wm_c = trowbridge_reitz_sample_wm(wo_local, sample_u, c_alpha_x, c_alpha_y)
                cos_θo_mc = dot(wo_local, wm_c)
                if cos_θo_mc < 0f0
                    return SpectralBSDFSample()
                end

                wi_local = -wo_local + 2f0 * cos_θo_mc * wm_c
                if wi_local[3] * wo_local[3] < 0f0
                    return SpectralBSDFSample()
                end

                F_conductor = fr_complex_spectral(abs(cos_θo_mc), ce_spectral, ck_spectral)

                D = trowbridge_reitz_d(wm_c, c_alpha_x, c_alpha_y)
                G = trowbridge_reitz_g(wo_local, wi_local, c_alpha_x, c_alpha_y)

                cos_i = abs(wi_local[3])
                cos_o = abs(wo_local[3])

                f_conductor = D * F_conductor * G / (4f0 * cos_i * cos_o)

                F_out = fresnel_dielectric(cos_i, ieta)
                T_out = 1f0 - F_out

                layer_tr = if has_medium
                    tr_in = layer_transmittance(thickness, Vec3f(0, 0, cos_o))
                    tr_out = layer_transmittance(thickness, wi_local)
                    tr_in * tr_out * albedo_spectral
                else
                    SpectralRadiance(1f0)
                end

                if flip
                    wi_local = -wi_local
                end

                wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
                wi = normalize(wi)

                f_spectral = f_conductor * T_in * T_out * layer_tr

                pdf_m = trowbridge_reitz_pdf(wo_local, wm_c, c_alpha_x, c_alpha_y)
                pdf = (1f0 - F_interface) * pdf_m / (4f0 * abs(cos_θo_mc))

                return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
            end
        end
    end
end

"""
    evaluate_bsdf_spectral(table, mat::CoatedConductor, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate CoatedConductor BSDF for given directions.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    mat::CoatedConductor, table::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths
)
    # Check hemisphere - coated conductor only reflects
    cos_θi = dot(wi, n)
    cos_θo = dot(wo, n)
    if cos_θi * cos_θo < 0f0
        return (SpectralRadiance(), 0f0)
    end

    abs_cos_θi = abs(cos_θi)
    abs_cos_θo = abs(cos_θo)

    if abs_cos_θi < 1f-6 || abs_cos_θo < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Get interface parameters
    ieta = mat.interface_eta
    if ieta == 0f0
        ieta = 1f0
    end

    iu_roughness = eval_tex(textures, mat.interface_u_roughness, tfc)
    iv_roughness = eval_tex(textures, mat.interface_v_roughness, tfc)
    i_alpha_x = mat.remap_roughness ? roughness_to_α(iu_roughness) : iu_roughness
    i_alpha_y = mat.remap_roughness ? roughness_to_α(iv_roughness) : iv_roughness

    # Get conductor parameters
    cu_roughness = eval_tex(textures, mat.conductor_u_roughness, tfc)
    cv_roughness = eval_tex(textures, mat.conductor_v_roughness, tfc)
    c_alpha_x = mat.remap_roughness ? roughness_to_α(cu_roughness) : cu_roughness
    c_alpha_y = mat.remap_roughness ? roughness_to_α(cv_roughness) : cv_roughness

    # Get conductor eta/k
    local ce_spectral::SpectralRadiance
    local ck_spectral::SpectralRadiance

    if mat.use_eta_k
        ce_spectral = eval_ior_spectral(table, textures, mat.conductor_eta, tfc, lambda)
        ck_spectral = eval_ior_spectral(table, textures, mat.conductor_k, tfc, lambda)
    else
        refl_rgb = eval_tex(textures, mat.reflectance, tfc)
        refl_rgb = RGBSpectrum(
            clamp(refl_rgb.c[1], 0f0, 0.9999f0),
            clamp(refl_rgb.c[2], 0f0, 0.9999f0),
            clamp(refl_rgb.c[3], 0f0, 0.9999f0)
        )
        r_spectral = uplift_rgb(table, refl_rgb, lambda)
        ce_spectral = SpectralRadiance(1f0)
        ck_spectral = 2f0 * sqrt(r_spectral) / sqrt(clamp_zero(SpectralRadiance(1f0) - r_spectral) + SpectralRadiance(1f-6))
    end

    # Scale by interface IOR
    ce_spectral = ce_spectral / ieta
    ck_spectral = ck_spectral / ieta

    # Volumetric parameters
    thickness = max(eval_tex(textures, mat.thickness, tfc), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, tfc)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)

    # Build local frame
    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), cos_θo)
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), cos_θi)

    flip = wo_local[3] < 0f0
    if flip
        wo_local = -wo_local
        wi_local = -wi_local
    end

    i_is_smooth = trowbridge_reitz_effectively_smooth(i_alpha_x, i_alpha_y)
    c_is_smooth = trowbridge_reitz_effectively_smooth(c_alpha_x, c_alpha_y)

    if i_is_smooth && c_is_smooth
        # Both smooth - delta functions, return zero for non-delta evaluation
        return (SpectralRadiance(), 0f0)
    end

    # Compute half-vector
    wh = normalize(wo_local + wi_local)
    if wh[3] < 0f0
        wh = -wh
    end
    cos_θo_h = dot(wo_local, wh)

    # Interface Fresnel
    F_interface_wh = fresnel_dielectric(abs(cos_θo_h), ieta)
    F_interface_o = fresnel_dielectric(abs(wo_local[3]), ieta)
    F_interface_i = fresnel_dielectric(abs(wi_local[3]), ieta)

    if i_is_smooth
        # Smooth interface, rough conductor
        # Only conductor contribution (interface specular is delta)
        T_o = 1f0 - F_interface_o
        T_i = 1f0 - F_interface_i

        c_alpha_x = max(c_alpha_x, 1f-4)
        c_alpha_y = max(c_alpha_y, 1f-4)

        D = trowbridge_reitz_d(wh, c_alpha_x, c_alpha_y)
        G = trowbridge_reitz_g(wo_local, wi_local, c_alpha_x, c_alpha_y)
        F_conductor = fr_complex_spectral(abs(cos_θo_h), ce_spectral, ck_spectral)

        f_conductor = D * F_conductor * G / (4f0 * abs(wi_local[3]) * abs(wo_local[3]))

        layer_tr = if has_medium
            tr = layer_transmittance(thickness, wi_local)
            tr * tr * albedo_spectral
        else
            SpectralRadiance(1f0)
        end

        f_spectral = f_conductor * T_o * T_i * layer_tr

        pdf_m = trowbridge_reitz_pdf(wo_local, wh, c_alpha_x, c_alpha_y)
        pdf = T_o * pdf_m / (4f0 * abs(cos_θo_h))

        return (f_spectral, pdf)
    else
        # Rough interface (and possibly rough conductor)
        i_alpha_x = max(i_alpha_x, 1f-4)
        i_alpha_y = max(i_alpha_y, 1f-4)

        # Interface specular contribution
        D_i = trowbridge_reitz_d(wh, i_alpha_x, i_alpha_y)
        G_i = trowbridge_reitz_g(wo_local, wi_local, i_alpha_x, i_alpha_y)
        f_interface = D_i * F_interface_wh * G_i / (4f0 * abs(wi_local[3]) * abs(wo_local[3]))

        # Conductor contribution
        T_o = 1f0 - F_interface_o
        T_i = 1f0 - F_interface_i

        local f_conductor::SpectralRadiance
        local pdf_conductor::Float32

        if c_is_smooth
            # Smooth conductor under rough interface
            F_conductor = fr_complex_spectral(abs(wo_local[3]), ce_spectral, ck_spectral)
            f_conductor = F_conductor / abs(wo_local[3])
            pdf_conductor = 1f0
        else
            c_alpha_x = max(c_alpha_x, 1f-4)
            c_alpha_y = max(c_alpha_y, 1f-4)

            D_c = trowbridge_reitz_d(wh, c_alpha_x, c_alpha_y)
            G_c = trowbridge_reitz_g(wo_local, wi_local, c_alpha_x, c_alpha_y)
            F_conductor = fr_complex_spectral(abs(cos_θo_h), ce_spectral, ck_spectral)

            f_conductor = D_c * F_conductor * G_c / (4f0 * abs(wi_local[3]) * abs(wo_local[3]))
            pdf_m_c = trowbridge_reitz_pdf(wo_local, wh, c_alpha_x, c_alpha_y)
            pdf_conductor = pdf_m_c / (4f0 * abs(cos_θo_h))
        end

        layer_tr = if has_medium
            tr = layer_transmittance(thickness, wi_local)
            tr * tr * albedo_spectral
        else
            SpectralRadiance(1f0)
        end

        f_conductor_contrib = f_conductor * T_o * T_i * layer_tr

        # Combined BSDF
        f_spectral = SpectralRadiance(f_interface) + f_conductor_contrib

        # Combined PDF
        pdf_m_i = trowbridge_reitz_pdf(wo_local, wh, i_alpha_x, i_alpha_y)
        pdf_interface = F_interface_o * pdf_m_i / (4f0 * abs(cos_θo_h))
        pdf = pdf_interface + T_o * pdf_conductor

        return (f_spectral, pdf)
    end
end
