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
    CETex, CKTex, CURoughTex, CVRoughTex,
    ThickTex, AlbedoTex, GTex
} <: Material
    # Interface parameters
    interface_u_roughness::IURoughTex  # Texture{Float32}
    interface_v_roughness::IVRoughTex  # Texture{Float32}
    interface_eta::Float32             # Scalar IOR for interface

    # Conductor parameters — always eta/k (reflectance is converted at construction time)
    conductor_eta::CETex               # Spectral complex IOR real part
    conductor_k::CKTex                 # Spectral complex IOR imaginary part
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
end

# Full constructor with all textures
function CoatedConductor(
    interface_u_roughness::Texture,
    interface_v_roughness::Texture,
    interface_eta::Float32,
    conductor_eta,  # Spectral or Texture — always required
    conductor_k,    # Spectral or Texture — always required
    conductor_u_roughness::Texture,
    conductor_v_roughness::Texture,
    thickness::Texture,
    albedo::Texture,
    g::Texture,
    max_depth::Int,
    n_samples::Int,
    remap_roughness::Bool
)
    CoatedConductor{
        typeof(interface_u_roughness), typeof(interface_v_roughness),
        typeof(conductor_eta), typeof(conductor_k),
        typeof(conductor_u_roughness), typeof(conductor_v_roughness),
        typeof(thickness), typeof(albedo), typeof(g)
    }(
        interface_u_roughness, interface_v_roughness, interface_eta,
        conductor_eta, conductor_k,
        conductor_u_roughness, conductor_v_roughness,
        thickness, albedo, g,
        Int32(max_depth), Int32(n_samples), remap_roughness,
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
    # Conductor parameters — eta/k OR reflectance (converted to eta/k at construction)
    conductor_eta = nothing,
    conductor_k = nothing,
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
    iu_rough, iv_rough = if interface_roughness isa Tuple
        Float32(interface_roughness[1]), Float32(interface_roughness[2])
    else
        Float32(interface_roughness), Float32(interface_roughness)
    end

    cu_rough, cv_rough = if conductor_roughness isa Tuple
        Float32(conductor_roughness[1]), Float32(conductor_roughness[2])
    else
        Float32(conductor_roughness), Float32(conductor_roughness)
    end

    # Always resolve to eta/k — matches pbrt-v4 materials.cpp:365-376
    local ce, ck
    if !isnothing(conductor_eta)
        isnothing(conductor_k) && error("conductor_k must be provided with conductor_eta")
        ce = conductor_eta
        ck = conductor_k
    elseif !isnothing(reflectance)
        # reflectance → eta/k conversion (pbrt-v4 lines 370-373)
        # eta = 1, k = 2*sqrt(r) / sqrt(1-r)
        # This is an RGB approximation — spectral conversion happens at eval time
        r = reflectance isa RGBSpectrum ? reflectance : RGBSpectrum(reflectance...)
        r = RGBSpectrum(clamp(r.c[1], 0f0, 0.9999f0), clamp(r.c[2], 0f0, 0.9999f0), clamp(r.c[3], 0f0, 0.9999f0))
        ce = RGBSpectrum(1f0)
        ck = RGBSpectrum(
            2f0 * sqrt(r.c[1]) / sqrt(max(1f-6, 1f0 - r.c[1])),
            2f0 * sqrt(r.c[2]) / sqrt(max(1f-6, 1f0 - r.c[2])),
            2f0 * sqrt(r.c[3]) / sqrt(max(1f-6, 1f0 - r.c[3])),
        )
    else
        # Default: copper (pbrt-v4 default for conductor)
        ce = RGBSpectrum(1f0)
        ck = RGBSpectrum(1f0)
    end

    CoatedConductor(
        to_texture(iu_rough), to_texture(iv_rough), Float32(interface_eta),
        to_texture(ce), to_texture(ck),
        to_texture(cu_rough), to_texture(cv_rough),
        to_texture(Float32(thickness)), to_texture(albedo), to_texture(Float32(g)),
        max_depth, n_samples, remap_roughness,
    )
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

    # Get conductor eta/k — always spectral (reflectance was converted at construction)
    # Matches pbrt-v4 materials.cpp:365-376
    ce_spectral = eval_ior_spectral(table, textures, mat.conductor_eta, tfc, lambda)
    ck_spectral = eval_ior_spectral(table, textures, mat.conductor_k, tfc, lambda)

    # Critical: scale conductor eta/k by interface IOR (pbrt-v4 lines 375-376)
    ce_spectral = ce_spectral / ieta
    ck_spectral = ck_spectral / ieta

    # Volumetric parameters
    thickness = max(eval_tex(textures, mat.thickness, tfc), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, tfc)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    g_val = clamp(eval_tex(textures, mat.g, tfc), -0.99f0, 0.99f0)
    has_medium = !is_black(albedo_rgb)

    max_depth = Int(mat.max_depth)

    # === LayeredBxDF random walk — identical to CoatedDiffuse but with conductor bottom ===
    # Matches pbrt-v4 bxdfs.h LayeredBxDF<DielectricBxDF, ConductorBxDF>::Sample_f

    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    flip_wi = wo_local[3] < 0f0
    if flip_wi
        wo_local = -wo_local
    end

    entered_top = true

    # Sample entrance interface (dielectric top)
    bs = sample_dielectric_interface(wo_local, rng, sample_u, i_alpha_x, i_alpha_y, ieta, BXDF_ALL)
    if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
        return SpectralBSDFSample()
    end

    # Entrance reflection → return immediately with pdfIsProportional
    if bs.is_reflection
        wi_local = bs.wi
        if flip_wi; wi_local = -wi_local; end
        wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
        wi = normalize(wi)
        flags = bs.is_specular ? BXDF_SPECULAR_REFLECTION : BXDF_GLOSSY_REFLECTION
        return SpectralBSDFSample(bs.f, wi, bs.pdf, flags, 1f0, true, false)
    end

    # Begin random walk
    w = bs.wi
    specular_path = bs.is_specular
    f = bs.f * abs(w[3])
    pdf = bs.pdf
    z = entered_top ? thickness : 0f0

    seed = UInt64(0)
    rng_state = pcg32_init(pbrt_hash(seed, wo_local), pbrt_hash(rng, sample_u))

    for depth in 0:(max_depth-1)
        rr_beta = max_component(f) / pdf
        if depth > 3 && rr_beta < 0.25f0
            q = max(0f0, 1f0 - rr_beta)
            rr_val, rng_state = pcg32_uniform_f32(rng_state)
            if rr_val < q; return SpectralBSDFSample(); end
            pdf *= 1f0 - q
        end
        w[3] == 0f0 && return SpectralBSDFSample()

        if has_medium
            sigma_t = 1f0
            exp_u, rng_state = pcg32_uniform_f32(rng_state)
            dz = sample_exponential(exp_u, sigma_t / abs(w[3]))
            zp = w[3] > 0f0 ? (z + dz) : (z - dz)
            zp == z && return SpectralBSDFSample()
            if 0f0 < zp && zp < thickness
                phase_u1, rng_state = pcg32_uniform_f32(rng_state)
                phase_u2, rng_state = pcg32_uniform_f32(rng_state)
                wi_phase, phase_p = sample_hg_phase_spectral(g_val, -w, Point2f(phase_u1, phase_u2))
                (phase_p == 0f0 || wi_phase[3] == 0f0) && return SpectralBSDFSample()
                f = f * albedo_spectral * phase_p
                pdf *= phase_p
                specular_path = false
                w = wi_phase
                z = zp
                continue
            end
            z = clamp(zp, 0f0, thickness)
        else
            z = (z == thickness) ? 0f0 : thickness
            f = f * layer_transmittance(thickness, w)
        end

        at_bottom = z == 0f0

        uc, rng_state = pcg32_uniform_f32(rng_state)
        u1, rng_state = pcg32_uniform_f32(rng_state)
        u2, rng_state = pcg32_uniform_f32(rng_state)
        u = Point2f(u1, u2)

        bs_interface = if at_bottom
            # Conductor bottom (reflection only)
            sample_conductor_interface(-w, u, c_alpha_x, c_alpha_y, ce_spectral, ck_spectral, BXDF_ALL)
        else
            # Dielectric top (can reflect or transmit)
            sample_dielectric_interface(-w, uc, u, i_alpha_x, i_alpha_y, ieta, BXDF_ALL)
        end

        if !bs_interface.valid || bs_interface.pdf == 0f0 || bs_interface.wi[3] == 0f0
            return SpectralBSDFSample()
        end

        f = f * bs_interface.f
        pdf *= bs_interface.pdf
        specular_path = specular_path && bs_interface.is_specular
        w = bs_interface.wi

        if !bs_interface.is_reflection
            wi_local = w
            if flip_wi; wi_local = -wi_local; end
            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)
            is_refl = same_hemisphere(wo_local, flip_wi ? -w : w)
            flags = if specular_path
                is_refl ? BXDF_SPECULAR_REFLECTION : BXDF_SPECULAR_TRANSMISSION
            else
                is_refl ? BXDF_GLOSSY_REFLECTION : BXDF_GLOSSY_TRANSMISSION
            end
            # pbrt-v4 hardcodes eta=1 for LayeredBxDF exit (bxdfs.h:768)
            return SpectralBSDFSample(f, wi, pdf, flags, 1f0, true, false)
        end

        f = f * abs(bs_interface.wi[3])
    end
    return SpectralBSDFSample()
end

"""
    evaluate_bsdf_spectral(table, mat::CoatedConductor, ...) -> (f, pdf)

Evaluate CoatedConductor BSDF using pbrt-v4's LayeredBxDF::f random walk.
Exact port — same as CoatedDiffuse evaluate but with conductor bottom interface.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    mat::CoatedConductor, table::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths
)
    ieta = mat.interface_eta
    ieta == 0f0 && (ieta = 1f0)
    iu_roughness = eval_tex(textures, mat.interface_u_roughness, tfc)
    iv_roughness = eval_tex(textures, mat.interface_v_roughness, tfc)
    i_alpha_x = mat.remap_roughness ? roughness_to_α(iu_roughness) : iu_roughness
    i_alpha_y = mat.remap_roughness ? roughness_to_α(iv_roughness) : iv_roughness
    cu_roughness = eval_tex(textures, mat.conductor_u_roughness, tfc)
    cv_roughness = eval_tex(textures, mat.conductor_v_roughness, tfc)
    c_alpha_x = mat.remap_roughness ? roughness_to_α(cu_roughness) : cu_roughness
    c_alpha_y = mat.remap_roughness ? roughness_to_α(cv_roughness) : cv_roughness

    ce_spectral = eval_ior_spectral(table, textures, mat.conductor_eta, tfc, lambda)
    ck_spectral = eval_ior_spectral(table, textures, mat.conductor_k, tfc, lambda)
    ce_spectral = ce_spectral / ieta
    ck_spectral = ck_spectral / ieta

    thickness = max(eval_tex(textures, mat.thickness, tfc), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, tfc)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)
    g_val = clamp(eval_tex(textures, mat.g, tfc), -0.99f0, 0.99f0)
    n_samples = Int(mat.n_samples)
    max_depth = Int(mat.max_depth)
    is_smooth = trowbridge_reitz_effectively_smooth(i_alpha_x, i_alpha_y)
    c_is_smooth = trowbridge_reitz_effectively_smooth(c_alpha_x, c_alpha_y)

    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, n))
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), dot(wi, n))
    if wo_local[3] < 0f0; wo_local = -wo_local; wi_local = -wi_local; end
    if abs(wo_local[3]) < 1f-6 || abs(wi_local[3]) < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    entered_top = true
    same_hemi = same_hemisphere(wo_local, wi_local)
    exit_at_bottom = same_hemi ⊻ entered_top
    exit_z = exit_at_bottom ? 0f0 : thickness
    # For CoatedConductor: bottom=conductor (never specular unless c_is_smooth), top=dielectric
    exit_is_specular = exit_at_bottom ? c_is_smooth : is_smooth
    nonexit_is_specular = exit_at_bottom ? is_smooth : c_is_smooth

    f_result = SpectralRadiance()
    if same_hemi
        enter_f, _ = eval_dielectric_interface(wo_local, wi_local, i_alpha_x, i_alpha_y, ieta)
        f_result = f_result + enter_f * Float32(n_samples)
    end

    rng = pcg32_init(pbrt_hash(UInt64(0), wo_local), pbrt_hash(wi_local))

    for s in 1:n_samples
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        wos = sample_dielectric_interface(wo_local, uc, Point2f(u1, u2), i_alpha_x, i_alpha_y, ieta, BXDF_TRANSMISSION)
        if !wos.valid || wos.pdf == 0f0 || wos.wi[3] == 0f0; continue; end

        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        # pbrt-v4: wis = exitInterface.Sample_f(wi, ..., !mode, Transmission)
        # !mode = Importance (no 1/etap² correction)
        wis = if exit_at_bottom
            sample_conductor_interface(wi_local, Point2f(u1, u2), c_alpha_x, c_alpha_y, ce_spectral, ck_spectral, BXDF_TRANSMISSION)
        else
            sample_dielectric_interface(wi_local, uc, Point2f(u1, u2), i_alpha_x, i_alpha_y, ieta, BXDF_TRANSMISSION, false)
        end
        if !wis.valid || wis.pdf == 0f0 || wis.wi[3] == 0f0; continue; end

        beta = wos.f * abs(wos.wi[3]) / wos.pdf
        z = entered_top ? thickness : 0f0
        w = wos.wi

        for depth in 0:(max_depth-1)
            if depth > 3 && max_component(beta) < 0.25f0
                q = max(0f0, 1f0 - max_component(beta))
                rr_val, rng = pcg32_uniform_f32(rng)
                if rr_val < q; break; end
                beta = beta / (1f0 - q)
            end

            if !has_medium
                z = (z == thickness) ? 0f0 : thickness
                beta = beta * layer_transmittance(thickness, w)
            else
                sigma_t = 1f0
                exp_u, rng = pcg32_uniform_f32(rng)
                dz = sample_exponential(exp_u, sigma_t / abs(w[3]))
                zp = w[3] > 0f0 ? (z + dz) : (z - dz)
                if zp == z; continue; end
                if 0f0 < zp && zp < thickness
                    wt = 1f0
                    if !exit_is_specular
                        wt = power_heuristic(1, wis.pdf, 1, hg_phase_pdf(g_val, dot(-w, -wis.wi)))
                    end
                    phase_val = hg_phase_pdf(g_val, dot(-w, -wis.wi))
                    f_result = f_result + beta * albedo_spectral * phase_val * wt *
                               layer_transmittance(zp - exit_z, wis.wi) * wis.f / wis.pdf

                    phase_u1, rng = pcg32_uniform_f32(rng)
                    phase_u2, rng = pcg32_uniform_f32(rng)
                    wi_phase, phase_p = sample_hg_phase_spectral(g_val, -w, Point2f(phase_u1, phase_u2))
                    if phase_p == 0f0 || wi_phase[3] == 0f0; continue; end
                    beta = beta * albedo_spectral * phase_p / phase_p
                    w = wi_phase; z = zp

                    if ((z < exit_z && w[3] > 0f0) || (z > exit_z && w[3] < 0f0)) && !exit_is_specular
                        f_exit, _ = if exit_at_bottom
                            eval_conductor_interface(-w, wi_local, c_alpha_x, c_alpha_y, ce_spectral, ck_spectral)
                        else
                            eval_dielectric_interface(-w, wi_local, i_alpha_x, i_alpha_y, ieta)
                        end
                        if max_component(f_exit) > 0f0
                            exit_pdf = if exit_at_bottom
                                pdf_conductor_interface(-w, wi_local, c_alpha_x, c_alpha_y)
                            else
                                pdf_dielectric_interface(-w, wi_local, i_alpha_x, i_alpha_y, ieta, BXDF_TRANSMISSION)
                            end
                            wt2 = power_heuristic(1, phase_p, 1, exit_pdf)
                            f_result = f_result + beta * layer_transmittance(zp - exit_z, wi_phase) * f_exit * wt2
                        end
                    end
                    continue
                end
                z = clamp(zp, 0f0, thickness)
            end

            if z == exit_z
                uc, rng = pcg32_uniform_f32(rng)
                u1, rng = pcg32_uniform_f32(rng)
                u2, rng = pcg32_uniform_f32(rng)
                bs = if exit_at_bottom
                    sample_conductor_interface(-w, Point2f(u1, u2), c_alpha_x, c_alpha_y, ce_spectral, ck_spectral, BXDF_REFLECTION)
                else
                    sample_dielectric_interface(-w, uc, Point2f(u1, u2), i_alpha_x, i_alpha_y, ieta, BXDF_REFLECTION)
                end
                if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0; break; end
                beta = beta * bs.f * abs(bs.wi[3]) / bs.pdf
                w = bs.wi
            else
                if !nonexit_is_specular
                    f_nee, _ = if z == thickness
                        eval_dielectric_interface(-w, -wis.wi, i_alpha_x, i_alpha_y, ieta)
                    else
                        eval_conductor_interface(-w, -wis.wi, c_alpha_x, c_alpha_y, ce_spectral, ck_spectral)
                    end
                    if max_component(f_nee) > 0f0
                        wt = 1f0
                        if !exit_is_specular
                            nee_pdf = if z == thickness
                                pdf_dielectric_interface(-w, -wis.wi, i_alpha_x, i_alpha_y, ieta)
                            else
                                pdf_conductor_interface(-w, -wis.wi, c_alpha_x, c_alpha_y)
                            end
                            wt = power_heuristic(1, wis.pdf, 1, nee_pdf)
                        end
                        f_result = f_result + beta * f_nee * abs(wis.wi[3]) * wt *
                                   layer_transmittance(thickness, wis.wi) * wis.f / wis.pdf
                    end
                end

                uc, rng = pcg32_uniform_f32(rng)
                u1, rng = pcg32_uniform_f32(rng)
                u2, rng = pcg32_uniform_f32(rng)
                bs = if z == thickness
                    sample_dielectric_interface(-w, uc, Point2f(u1, u2), i_alpha_x, i_alpha_y, ieta, BXDF_REFLECTION)
                else
                    sample_conductor_interface(-w, Point2f(u1, u2), c_alpha_x, c_alpha_y, ce_spectral, ck_spectral, BXDF_REFLECTION)
                end
                if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0; break; end
                beta = beta * bs.f * abs(bs.wi[3]) / bs.pdf
                w = bs.wi

                if !exit_is_specular
                    f_exit, _ = if exit_at_bottom
                        eval_conductor_interface(-w, wi_local, c_alpha_x, c_alpha_y, ce_spectral, ck_spectral)
                    else
                        eval_dielectric_interface(-w, wi_local, i_alpha_x, i_alpha_y, ieta)
                    end
                    if max_component(f_exit) > 0f0
                        wt3 = 1f0
                        if !nonexit_is_specular
                            exit_pdf3 = if exit_at_bottom
                                pdf_conductor_interface(-w, wi_local, c_alpha_x, c_alpha_y)
                            else
                                pdf_dielectric_interface(-w, wi_local, i_alpha_x, i_alpha_y, ieta, BXDF_TRANSMISSION)
                            end
                            wt3 = power_heuristic(1, bs.pdf, 1, exit_pdf3)
                        end
                        f_result = f_result + beta * layer_transmittance(thickness, bs.wi) * f_exit * wt3
                    end
                end
            end
        end
    end

    f_result = f_result / Float32(n_samples)
    pdf = pdf_layered_conductor_bsdf(wo_local, wi_local, i_alpha_x, i_alpha_y, ieta, c_alpha_x, c_alpha_y, ce_spectral, ck_spectral, n_samples, max_depth, has_medium, g_val, thickness)
    return (f_result, pdf)
end

"""PDF for LayeredBxDF<Dielectric, Conductor>. Matches pbrt-v4 LayeredBxDF::PDF."""
@propagate_inbounds function pdf_layered_conductor_bsdf(
    wo::Vec3f, wi::Vec3f,
    i_alpha_x::Float32, i_alpha_y::Float32, ieta::Float32,
    c_alpha_x::Float32, c_alpha_y::Float32,
    ce::SpectralRadiance, ck::SpectralRadiance,
    n_samples::Int, max_depth::Int,
    has_medium::Bool, g_val::Float32, thickness::Float32
)
    rng = pcg32_init(pbrt_hash(UInt64(0), wi), pbrt_hash(wo))
    is_smooth = trowbridge_reitz_effectively_smooth(i_alpha_x, i_alpha_y)
    c_is_smooth = trowbridge_reitz_effectively_smooth(c_alpha_x, c_alpha_y)
    same_hemi = same_hemisphere(wo, wi)

    pdf_sum = 0f0
    if same_hemi
        pdf_sum += Float32(n_samples) * (is_smooth ? 0f0 : pdf_dielectric_interface(wo, wi, i_alpha_x, i_alpha_y, ieta, BXDF_REFLECTION))
    end

    for s in 1:n_samples
        if same_hemi
            uc1, rng = pcg32_uniform_f32(rng)
            u1, rng = pcg32_uniform_f32(rng)
            u2, rng = pcg32_uniform_f32(rng)
            wos = sample_dielectric_interface(wo, uc1, Point2f(u1, u2), i_alpha_x, i_alpha_y, ieta, BXDF_TRANSMISSION)
            uc2, rng = pcg32_uniform_f32(rng)
            u3, rng = pcg32_uniform_f32(rng)
            u4, rng = pcg32_uniform_f32(rng)
            wis = sample_dielectric_interface(wi, uc2, Point2f(u3, u4), i_alpha_x, i_alpha_y, ieta, BXDF_TRANSMISSION)

            if wos.valid && wos.pdf > 0f0 && wis.valid && wis.pdf > 0f0
                if is_smooth
                    # Specular top: use bottom (conductor) PDF
                    pdf_sum += c_is_smooth ? 0f0 : pdf_conductor_interface(-wos.wi, -wis.wi, c_alpha_x, c_alpha_y)
                else
                    u5, rng = pcg32_uniform_f32(rng)
                    u6, rng = pcg32_uniform_f32(rng)
                    rs = sample_conductor_interface(-wos.wi, Point2f(u5, u6), c_alpha_x, c_alpha_y, ce, ck, BXDF_ALL)
                    if rs.valid && rs.pdf > 0f0
                        if c_is_smooth
                            pdf_sum += pdf_dielectric_interface(-rs.wi, wi, i_alpha_x, i_alpha_y, ieta)
                        else
                            r_pdf = pdf_conductor_interface(-wos.wi, -wis.wi, c_alpha_x, c_alpha_y)
                            wt = power_heuristic(1, wis.pdf, 1, r_pdf)
                            pdf_sum += wt * r_pdf
                            t_pdf = pdf_dielectric_interface(-rs.wi, wi, i_alpha_x, i_alpha_y, ieta)
                            wt2 = power_heuristic(1, rs.pdf, 1, t_pdf)
                            pdf_sum += wt2 * t_pdf
                        end
                    end
                end
            end
        else
            # TT term — CoatedConductor is reflection-only, so this shouldn't happen
            # but included for completeness matching pbrt's structure
            continue
        end
    end

    return lerp(0.9f0, 1f0 / (4f0 * Float32(π)), pdf_sum / Float32(n_samples))
end
