# ============================================================================
# CoatedDiffuseTransmission - Layered: dielectric coating + diffuse transmission
# ============================================================================
# Extension of CoatedDiffuse that uses DiffuseTransmissionBxDF as the
# bottom layer instead of DiffuseBxDF. This creates a material with a glossy
# dielectric coating over a diffuse substrate that both reflects and transmits.
#
# Ideal for: leaves, thin fabric, paper with a glossy surface coating.
#
# In pbrt-v4 terms: LayeredBxDF<DielectricBxDF, DiffuseTransmissionBxDF, true>

struct CoatedDiffuseTransmission{ReflTex, TransTex, URoughTex, VRoughTex, ThickTex, AlbedoTex, GTex} <: Material
    reflectance::ReflTex    # Diffuse reflectance (same hemisphere)
    transmittance::TransTex # Diffuse transmittance (opposite hemisphere)
    u_roughness::URoughTex
    v_roughness::VRoughTex
    thickness::ThickTex     # Coating layer thickness
    eta::Float32            # IOR of dielectric coating
    albedo::AlbedoTex       # Medium albedo between layers (0 = no medium)
    g::GTex                 # HG asymmetry parameter
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
end

# Full constructor
function CoatedDiffuseTransmission(
    reflectance::Texture,
    transmittance::Texture,
    u_roughness::Texture,
    v_roughness::Texture,
    thickness::Texture,
    eta::Float32,
    albedo::Texture,
    g::Texture,
    max_depth::Int,
    n_samples::Int,
    remap_roughness::Bool
)
    CoatedDiffuseTransmission{
        typeof(reflectance), typeof(transmittance),
        typeof(u_roughness), typeof(v_roughness),
        typeof(thickness), typeof(albedo), typeof(g)
    }(
        reflectance, transmittance, u_roughness, v_roughness, thickness,
        eta, albedo, g, Int32(max_depth), Int32(n_samples), remap_roughness
    )
end

function CoatedDiffuseTransmission(;
    reflectance = RGBSpectrum(0.5f0),
    transmittance = RGBSpectrum(0.25f0),
    roughness = 0f0,
    thickness = 0.01f0,
    eta::Real = 1.5f0,
    albedo = RGBSpectrum(0f0),
    g = 0f0,
    max_depth::Int = 10,
    n_samples::Int = 1,
    remap_roughness::Bool = true
)
    u_rough, v_rough = if roughness isa Tuple
        Float32(roughness[1]), Float32(roughness[2])
    else
        Float32(roughness), Float32(roughness)
    end

    CoatedDiffuseTransmission(
        to_texture(reflectance),
        to_texture(transmittance),
        to_texture(u_rough),
        to_texture(v_rough),
        to_texture(Float32(thickness)),
        Float32(eta),
        to_texture(albedo),
        to_texture(Float32(g)),
        max_depth,
        n_samples,
        remap_roughness
    )
end

is_emissive(::CoatedDiffuseTransmission) = false


# ============================================================================
# Spectral BSDF Evaluation — CoatedDiffuseTransmission
# ============================================================================
# Same LayeredBxDF random walk as CoatedDiffuse, but the bottom layer uses
# DiffuseTransmissionBxDF (reflects + transmits) instead of DiffuseBxDF (reflects only).

# --- Bottom interface: DiffuseTransmission ---

"""
    sample_diffuse_transmission_bottom(wo, u, uc, reflectance, transmittance, refl_trans_flags) -> LayeredBSDFSample

Sample the diffuse transmission bottom layer for the LayeredBxDF walk.
Handles both reflection (R/π, same hemisphere) and transmission (T/π, opposite hemisphere).
"""
@propagate_inbounds function sample_diffuse_transmission_bottom(
    wo::Vec3f, u::Point2f, uc::Float32,
    reflectance::SpectralRadiance, transmittance::SpectralRadiance,
    pr_max::Float32, pt_max::Float32,
    refl_trans_flags::UInt8
)
    pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? pr_max : 0f0
    pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? pt_max : 0f0

    if pr + pt < 1f-10
        return LayeredBSDFSample()
    end

    prob_reflect = pr / (pr + pt)

    wi = cosine_sample_hemisphere(u)

    if uc < prob_reflect
        # Reflection: same hemisphere as wo
        if wo[3] < 0f0
            wi = Vec3f(wi[1], wi[2], -wi[3])
        end
        cos_θi = abs(wi[3])
        cos_θi < 1f-6 && return LayeredBSDFSample()
        f = reflectance * (1f0 / Float32(π))
        pdf = prob_reflect * cos_θi / Float32(π)
        return LayeredBSDFSample(f, wi, pdf, true, false, 1f0, true)
    else
        # Transmission: opposite hemisphere from wo
        if wo[3] > 0f0
            wi = Vec3f(wi[1], wi[2], -wi[3])
        end
        cos_θi = abs(wi[3])
        cos_θi < 1f-6 && return LayeredBSDFSample()
        f = transmittance * (1f0 / Float32(π))
        pdf = (1f0 - prob_reflect) * cos_θi / Float32(π)
        return LayeredBSDFSample(f, wi, pdf, false, false, 1f0, true)
    end
end

@propagate_inbounds function eval_diffuse_transmission_bottom(
    wo::Vec3f, wi::Vec3f,
    reflectance::SpectralRadiance, transmittance::SpectralRadiance,
    pr_max::Float32, pt_max::Float32
)
    if pr_max + pt_max < 1f-10
        return (SpectralRadiance(), 0f0)
    end
    abs_cos_θi = abs(wi[3])
    if same_hemisphere(wo, wi)
        f = reflectance * (1f0 / Float32(π))
        prob_r = pr_max / (pr_max + pt_max)
        pdf = prob_r * abs_cos_θi / Float32(π)
        return (f, pdf)
    else
        f = transmittance * (1f0 / Float32(π))
        prob_t = pt_max / (pr_max + pt_max)
        pdf = prob_t * abs_cos_θi / Float32(π)
        return (f, pdf)
    end
end

@propagate_inbounds function pdf_diffuse_transmission_bottom(
    wo::Vec3f, wi::Vec3f,
    pr_max::Float32, pt_max::Float32,
    refl_trans_flags::UInt8 = BXDF_ALL
)
    pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? pr_max : 0f0
    pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? pt_max : 0f0

    if pr + pt < 1f-10
        return 0f0
    end

    abs_cos_θi = abs(wi[3])
    if same_hemisphere(wo, wi)
        return (pr / (pr + pt)) * abs_cos_θi / Float32(π)
    else
        return (pt / (pr + pt)) * abs_cos_θi / Float32(π)
    end
end

# --- sample_bsdf_spectral for CoatedDiffuseTransmission ---

@propagate_inbounds function sample_bsdf_spectral(
    mat::CoatedDiffuseTransmission, table::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, tfc::TextureFilterContext,
    lambda::Wavelengths, sample_u::Point2f, rng_in::Float32,
    regularize::Bool = false
)
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties
    refl_rgb = eval_tex(textures, mat.reflectance, tfc)
    trans_rgb = eval_tex(textures, mat.transmittance, tfc)
    eta = mat.eta
    thickness = max(eval_tex(textures, mat.thickness, tfc), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, tfc)
    g_val = clamp(eval_tex(textures, mat.g, tfc), -0.99f0, 0.99f0)

    u_roughness = eval_tex(textures, mat.u_roughness, tfc)
    v_roughness = eval_tex(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end

    # Clamp and uplift
    refl_rgb = RGBSpectrum(clamp(refl_rgb.c[1], 0f0, 1f0), clamp(refl_rgb.c[2], 0f0, 1f0), clamp(refl_rgb.c[3], 0f0, 1f0))
    trans_rgb = RGBSpectrum(clamp(trans_rgb.c[1], 0f0, 1f0), clamp(trans_rgb.c[2], 0f0, 1f0), clamp(trans_rgb.c[3], 0f0, 1f0))
    refl_spectral = uplift_rgb(table, refl_rgb, lambda)
    trans_spectral = uplift_rgb(table, trans_rgb, lambda)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)

    # Bottom layer probabilities (for sampling reflection vs transmission)
    pr_max = max(refl_rgb.c[1], refl_rgb.c[2], refl_rgb.c[3])
    pt_max = max(trans_rgb.c[1], trans_rgb.c[2], trans_rgb.c[3])

    max_depth = Int(mat.max_depth)

    # Build local frame
    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    # Two-sided: flip if entering from below
    flip_wi = wo_local[3] < 0f0
    if flip_wi
        wo_local = -wo_local
    end

    # Sample entrance (dielectric top)
    bs = sample_dielectric_interface(wo_local, rng_in, sample_u, alpha_x, alpha_y, eta, BXDF_ALL)
    if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
        return SpectralBSDFSample()
    end

    # Reflection at entrance: return immediately
    if bs.is_reflection
        wi_local = flip_wi ? -bs.wi : bs.wi
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
    z = thickness  # After transmitting through top, we're at bottom of coating

    rng = pcg32_init(pbrt_hash(UInt64(0), wo_local), pbrt_hash(rng_in, sample_u))

    for depth in 0:(max_depth-1)
        # Russian Roulette
        rr_beta = max_component(f) / pdf
        if depth > 3 && rr_beta < 0.25f0
            q = max(0f0, 1f0 - rr_beta)
            rr_val, rng = pcg32_uniform_f32(rng)
            if rr_val < q
                return SpectralBSDFSample()
            end
            pdf *= 1f0 - q
        end

        w[3] == 0f0 && return SpectralBSDFSample()

        if has_medium
            sigma_t = 1f0
            exp_u, rng = pcg32_uniform_f32(rng)
            dz = sample_exponential(exp_u, sigma_t / abs(w[3]))
            zp = w[3] > 0f0 ? (z + dz) : (z - dz)
            zp == z && return SpectralBSDFSample()

            if 0f0 < zp && zp < thickness
                # Medium scattering event
                phase_u1, rng = pcg32_uniform_f32(rng)
                phase_u2, rng = pcg32_uniform_f32(rng)
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

        # Sample interface
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        u = Point2f(u1, u2)

        at_bottom = z == 0f0

        bs_interface = if at_bottom
            # Bottom: diffuse transmission (can reflect or transmit through)
            sample_diffuse_transmission_bottom(-w, u, uc, refl_spectral, trans_spectral, pr_max, pt_max, BXDF_ALL)
        else
            # Top: dielectric (can reflect or transmit)
            sample_dielectric_interface(-w, uc, u, alpha_x, alpha_y, eta, BXDF_ALL)
        end

        if !bs_interface.valid || bs_interface.pdf == 0f0 || bs_interface.wi[3] == 0f0
            return SpectralBSDFSample()
        end

        f = f * bs_interface.f
        pdf *= bs_interface.pdf
        specular_path = specular_path && bs_interface.is_specular
        w = bs_interface.wi

        # Check if ray exited the layers
        if !bs_interface.is_reflection
            wi_local = flip_wi ? -w : w
            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)
            is_refl = same_hemisphere(wo_local, flip_wi ? -w : w)
            flags = if specular_path
                is_refl ? BXDF_SPECULAR_REFLECTION : BXDF_SPECULAR_TRANSMISSION
            else
                is_refl ? BXDF_GLOSSY_REFLECTION : BXDF_GLOSSY_TRANSMISSION
            end
            return SpectralBSDFSample(f, wi, pdf, flags, bs_interface.eta, true, false)
        end

        # Continue walk
        f = f * abs(bs_interface.wi[3])
    end

    return SpectralBSDFSample()
end

# --- evaluate_bsdf_spectral for CoatedDiffuseTransmission ---

@propagate_inbounds function evaluate_bsdf_spectral(
    mat::CoatedDiffuseTransmission, table::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, tfc::TextureFilterContext, lambda::Wavelengths
)
    # Get material properties
    refl_rgb = eval_tex(textures, mat.reflectance, tfc)
    trans_rgb = eval_tex(textures, mat.transmittance, tfc)
    eta = mat.eta
    thickness = max(eval_tex(textures, mat.thickness, tfc), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, tfc)
    g_val = clamp(eval_tex(textures, mat.g, tfc), -0.99f0, 0.99f0)

    u_roughness = eval_tex(textures, mat.u_roughness, tfc)
    v_roughness = eval_tex(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness

    refl_rgb = RGBSpectrum(clamp(refl_rgb.c[1], 0f0, 1f0), clamp(refl_rgb.c[2], 0f0, 1f0), clamp(refl_rgb.c[3], 0f0, 1f0))
    trans_rgb = RGBSpectrum(clamp(trans_rgb.c[1], 0f0, 1f0), clamp(trans_rgb.c[2], 0f0, 1f0), clamp(trans_rgb.c[3], 0f0, 1f0))
    refl_spectral = uplift_rgb(table, refl_rgb, lambda)
    trans_spectral = uplift_rgb(table, trans_rgb, lambda)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)

    pr_max = max(refl_rgb.c[1], refl_rgb.c[2], refl_rgb.c[3])
    pt_max = max(trans_rgb.c[1], trans_rgb.c[2], trans_rgb.c[3])

    n_samples = Int(mat.n_samples)
    max_depth = Int(mat.max_depth)

    # Build local frame
    tangent, bitangent = coordinate_system(n)
    cos_θo = dot(wo, n)
    cos_θi = dot(wi, n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), cos_θo)
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), cos_θi)

    # Two-sided flip
    if wo_local[3] < 0f0
        wo_local = -wo_local
        wi_local = -wi_local
    end

    if abs(wo_local[3]) < 1f-6 || abs(wi_local[3]) < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    entered_top = true
    same_hemi = same_hemisphere(wo_local, wi_local)

    # Exit interface determination (pbrt-v4 LayeredBxDF logic)
    # SameHemisphere(wo,wi) ^ enteredTop: if true→exit at bottom, else→exit at top
    exit_at_bottom = same_hemi ⊻ entered_top
    exit_z = exit_at_bottom ? 0f0 : thickness

    f_result = SpectralRadiance()

    # Direct reflection at entrance (same hemisphere only)
    if same_hemi
        enter_f, _ = eval_dielectric_interface(wo_local, wi_local, alpha_x, alpha_y, eta)
        f_result = f_result + enter_f * Float32(n_samples)
    end

    rng = pcg32_init(pbrt_hash(UInt64(0), wo_local), pbrt_hash(wi_local))
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    for s in 1:n_samples
        # Sample transmission through entrance (wo direction)
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        wos = sample_dielectric_interface(wo_local, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
        if !wos.valid || wos.pdf == 0f0 || wos.wi[3] == 0f0
            continue
        end

        # Sample "virtual light" from exit interface (wi direction)
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        wis = if exit_at_bottom
            sample_diffuse_transmission_bottom(wi_local, Point2f(u1, u2), uc, refl_spectral, trans_spectral, pr_max, pt_max, BXDF_TRANSMISSION)
        else
            sample_dielectric_interface(wi_local, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
        end
        if !wis.valid || wis.pdf == 0f0 || wis.wi[3] == 0f0
            continue
        end

        # Random walk
        beta = wos.f * abs(wos.wi[3]) / wos.pdf
        z = entered_top ? thickness : 0f0
        w = wos.wi

        for depth in 0:(max_depth-1)
            if depth > 3 && max_component(beta) < 0.25f0
                q = max(0f0, 1f0 - max_component(beta))
                rr_val, rng = pcg32_uniform_f32(rng)
                if rr_val < q
                    break
                end
                beta = beta / (1f0 - q)
            end

            if has_medium
                sigma_t = 1f0
                exp_u, rng = pcg32_uniform_f32(rng)
                dz = sample_exponential(exp_u, sigma_t / abs(w[3]))
                zp = w[3] > 0f0 ? (z + dz) : (z - dz)
                zp == z && continue

                if 0f0 < zp && zp < thickness
                    # Medium scattering - NEE through exit
                    if exit_at_bottom
                        wt = power_heuristic(1, wis.pdf, 1, hg_phase_pdf(g_val, dot(-w, -wis.wi)))
                        f_exit, _ = eval_diffuse_transmission_bottom(-w, -wis.wi, refl_spectral, trans_spectral, pr_max, pt_max)
                    else
                        if !is_smooth
                            wt = power_heuristic(1, wis.pdf, 1, hg_phase_pdf(g_val, dot(-w, -wis.wi)))
                        else
                            wt = 1f0
                        end
                        f_exit, _ = eval_dielectric_interface(-w, -wis.wi, alpha_x, alpha_y, eta)
                    end

                    phase_val = hg_phase_pdf(g_val, dot(-w, -wis.wi))
                    f_result = f_result + beta * albedo_spectral * phase_val * wt *
                               layer_transmittance(zp - exit_z, wis.wi) * wis.f / wis.pdf

                    # Sample phase function
                    phase_u1, rng = pcg32_uniform_f32(rng)
                    phase_u2, rng = pcg32_uniform_f32(rng)
                    wi_phase, phase_p = sample_hg_phase_spectral(g_val, -w, Point2f(phase_u1, phase_u2))
                    (phase_p == 0f0 || wi_phase[3] == 0f0) && break

                    beta = beta * albedo_spectral * phase_p / phase_p
                    w = wi_phase
                    z = zp

                    # NEE through exit after phase scatter
                    if ((z < exit_z && w[3] > 0f0) || (z > exit_z && w[3] < 0f0))
                        if exit_at_bottom
                            f_exit2, exit_pdf = eval_diffuse_transmission_bottom(-w, wi_local, refl_spectral, trans_spectral, pr_max, pt_max)
                        else
                            if !is_smooth
                                f_exit2, _ = eval_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta)
                                exit_pdf = pdf_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
                            else
                                continue
                            end
                        end
                        if max_component(f_exit2) > 0f0
                            wt2 = power_heuristic(1, phase_p, 1, exit_pdf)
                            f_result = f_result + beta * layer_transmittance(zp - exit_z, wi_phase) * f_exit2 * wt2
                        end
                    end

                    continue
                end
                z = clamp(zp, 0f0, thickness)
            else
                z = (z == thickness) ? 0f0 : thickness
                beta = beta * layer_transmittance(thickness, w)
            end

            at_exit = z == exit_z

            if at_exit
                # At exit: sample reflection to continue walk
                uc, rng = pcg32_uniform_f32(rng)
                u1, rng = pcg32_uniform_f32(rng)
                u2, rng = pcg32_uniform_f32(rng)
                bs = if exit_at_bottom
                    sample_diffuse_transmission_bottom(-w, Point2f(u1, u2), uc, refl_spectral, trans_spectral, pr_max, pt_max, BXDF_REFLECTION)
                else
                    sample_dielectric_interface(-w, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_REFLECTION)
                end
                (!bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0) && break
                beta = beta * bs.f * abs(bs.wi[3]) / bs.pdf
                w = bs.wi
            else
                # At non-exit: NEE + sample reflection
                non_exit_is_bottom = (z == 0f0)
                non_exit_is_specular = !non_exit_is_bottom && is_smooth  # bottom is never specular

                if !non_exit_is_specular
                    if non_exit_is_bottom
                        f_nee, _ = eval_diffuse_transmission_bottom(-w, -wis.wi, refl_spectral, trans_spectral, pr_max, pt_max)
                    else
                        f_nee, _ = eval_dielectric_interface(-w, -wis.wi, alpha_x, alpha_y, eta)
                    end

                    if max_component(f_nee) > 0f0
                        wt = 1f0
                        if !exit_at_bottom || !is_smooth
                            if non_exit_is_bottom
                                nee_pdf = pdf_diffuse_transmission_bottom(-w, -wis.wi, pr_max, pt_max)
                            else
                                nee_pdf = pdf_dielectric_interface(-w, -wis.wi, alpha_x, alpha_y, eta)
                            end
                            wt = power_heuristic(1, wis.pdf, 1, nee_pdf)
                        end
                        f_result = f_result + beta * f_nee * abs(wis.wi[3]) * wt *
                                   layer_transmittance(thickness, wis.wi) * wis.f / wis.pdf
                    end
                end

                # Sample new direction
                uc, rng = pcg32_uniform_f32(rng)
                u1, rng = pcg32_uniform_f32(rng)
                u2, rng = pcg32_uniform_f32(rng)
                bs = if non_exit_is_bottom
                    sample_diffuse_transmission_bottom(-w, Point2f(u1, u2), uc, refl_spectral, trans_spectral, pr_max, pt_max, BXDF_REFLECTION)
                else
                    sample_dielectric_interface(-w, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_REFLECTION)
                end
                (!bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0) && break

                beta = beta * bs.f * abs(bs.wi[3]) / bs.pdf
                w = bs.wi

                # NEE through exit after scattering
                if !is_smooth || exit_at_bottom
                    if exit_at_bottom
                        f_exit3, _ = eval_diffuse_transmission_bottom(-w, wi_local, refl_spectral, trans_spectral, pr_max, pt_max)
                    else
                        f_exit3, _ = eval_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta)
                    end

                    if max_component(f_exit3) > 0f0
                        wt3 = 1f0
                        if !non_exit_is_specular
                            if exit_at_bottom
                                exit_pdf3 = pdf_diffuse_transmission_bottom(-w, wi_local, pr_max, pt_max)
                            else
                                exit_pdf3 = pdf_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
                            end
                            wt3 = power_heuristic(1, bs.pdf, 1, exit_pdf3)
                        end
                        f_result = f_result + beta * layer_transmittance(thickness, bs.wi) * f_exit3 * wt3
                    end
                end
            end
        end
    end

    f_result = f_result / Float32(n_samples)

    # PDF estimation
    pdf = pdf_layered_bsdf_dt(wo_local, wi_local, alpha_x, alpha_y, eta, n_samples, max_depth,
                               refl_spectral, trans_spectral, pr_max, pt_max, has_medium, g_val, thickness)

    return (f_result, pdf)
end

# --- PDF estimation for CoatedDiffuseTransmission ---

@propagate_inbounds function pdf_layered_bsdf_dt(
    wo::Vec3f, wi::Vec3f,
    alpha_x::Float32, alpha_y::Float32, eta::Float32,
    n_samples::Int, max_depth::Int,
    refl_spectral::SpectralRadiance, trans_spectral::SpectralRadiance,
    pr_max::Float32, pt_max::Float32,
    has_medium::Bool, g_val::Float32, thickness::Float32
)
    rng = pcg32_init(pbrt_hash(UInt64(0), wi), pbrt_hash(wo))
    same_hemi = same_hemisphere(wo, wi)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    pdf_sum = 0f0

    # Entrance reflection PDF (same hemisphere only)
    if same_hemi
        if !is_smooth
            pdf_sum += Float32(n_samples) * pdf_dielectric_interface(wo, wi, alpha_x, alpha_y, eta, BXDF_REFLECTION)
        end
    end

    for s in 1:n_samples
        if same_hemi
            # TRT term: top→bottom(reflect)→top
            uc1, rng = pcg32_uniform_f32(rng)
            u1, rng = pcg32_uniform_f32(rng)
            u2, rng = pcg32_uniform_f32(rng)
            wos = sample_dielectric_interface(wo, uc1, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)

            uc2, rng = pcg32_uniform_f32(rng)
            u3, rng = pcg32_uniform_f32(rng)
            u4, rng = pcg32_uniform_f32(rng)
            wis = sample_dielectric_interface(wi, uc2, Point2f(u3, u4), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)

            if wos.valid && wos.pdf > 0f0 && wis.valid && wis.pdf > 0f0
                if is_smooth
                    pdf_sum += pdf_diffuse_transmission_bottom(-wos.wi, -wis.wi, pr_max, pt_max)
                else
                    u5, rng = pcg32_uniform_f32(rng)
                    u6, rng = pcg32_uniform_f32(rng)
                    uc3, rng = pcg32_uniform_f32(rng)
                    rs = sample_diffuse_transmission_bottom(-wos.wi, Point2f(u5, u6), uc3, refl_spectral, trans_spectral, pr_max, pt_max, BXDF_ALL)
                    if rs.valid && rs.pdf > 0f0
                        r_pdf = pdf_diffuse_transmission_bottom(-wos.wi, -wis.wi, pr_max, pt_max)
                        wt = power_heuristic(1, wis.pdf, 1, r_pdf)
                        pdf_sum += wt * r_pdf

                        t_pdf = pdf_dielectric_interface(-rs.wi, wi, alpha_x, alpha_y, eta)
                        wt2 = power_heuristic(1, rs.pdf, 1, t_pdf)
                        pdf_sum += wt2 * t_pdf
                    end
                end
            end
        else
            # TT term: top→bottom(transmit)
            uc1, rng = pcg32_uniform_f32(rng)
            u1, rng = pcg32_uniform_f32(rng)
            u2, rng = pcg32_uniform_f32(rng)
            wos = sample_dielectric_interface(wo, uc1, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
            if !wos.valid || wos.pdf == 0f0 || wos.is_reflection
                continue
            end

            # For transmission through bottom: the exit interface is the bottom (diffuse transmission)
            uc2, rng = pcg32_uniform_f32(rng)
            u3, rng = pcg32_uniform_f32(rng)
            u4, rng = pcg32_uniform_f32(rng)
            wis = sample_diffuse_transmission_bottom(wi, Point2f(u3, u4), uc2, refl_spectral, trans_spectral, pr_max, pt_max, BXDF_TRANSMISSION)
            if !wis.valid || wis.pdf == 0f0 || wis.is_reflection
                continue
            end

            if is_smooth
                pdf_sum += pdf_diffuse_transmission_bottom(-wos.wi, wi, pr_max, pt_max)
            else
                pdf_sum += (pdf_dielectric_interface(wo, -wis.wi, alpha_x, alpha_y, eta) +
                           pdf_diffuse_transmission_bottom(-wos.wi, wi, pr_max, pt_max)) / 2f0
            end
        end
    end

    return lerp(0.9f0, 1f0 / (4f0 * Float32(π)), pdf_sum / Float32(n_samples))
end
