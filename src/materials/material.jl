struct MatteMaterial <: Material
    """
    Spectral diffese reflection value.
    """
    Kd::Texture  # TODO check that texture is spectral
    """
    Scalar roughness.
    """
    σ::Texture  # TODO check that texture is scalar
    # TODO bump map
end

"""
Compute scattering function.
"""
function (m::MatteMaterial)(
    si::SurfaceInteraction, allow_multiple_lobes::Bool, ::Type{T},
) where T <: TransportMode
    # TODO perform bump mapping
    # Evaluate textures and create BSDF.
    si.bsdf = si |> BSDF
    r = si |> m.Kd |> clamp
    is_black(r) && return

    σ = clamp(si |> m.σ, 0f0, 90f0)
    if σ ≈ 0f0
        add!(si.bsdf, LambertianReflection(r))
    else
        add!(si.bsdf, OrenNayar(r, σ))
    end
end


struct MirrorMaterial <: Material
    Kr::Texture
    # TODO bump map
end

function (m::MirrorMaterial)(
    si::SurfaceInteraction, allow_multiple_lobes::Bool, ::Type{T},
) where T <: TransportMode
    si.bsdf = si |> BSDF
    r = si |> m.Kr |> clamp
    is_black(r) && return
    add!(si.bsdf, SpecularReflection(r, FresnelNoOp()))
end


struct GlassMaterial <: Material
    """
    Spectrum texture.
    """
    Kr::Texture
    """
    Spectrum texture.
    """
    Kt::Texture
    """
    Float texture.
    """
    u_roughness::Texture
    """
    Float texture.
    """
    v_roughness::Texture
    """
    Float texture.
    """
    index::Texture

    remap_roughness::Bool
    # TODO bump mapping
end

function (g::GlassMaterial)(
    si::SurfaceInteraction, allow_multiple_lobes::Bool, ::Type{T},
) where T <: TransportMode
    η = si |> g.index
    u_roughness = si |> g.u_roughness
    v_roughness = si |> g.v_roughness
    # println("=== Glass Material ===")
    # println("\t- η $η")
    # println("\t- UV roughness $u_roughness, $v_roughness")

    si.bsdf = BSDF(si, η)
    r = si |> g.Kr |> clamp
    t = si |> g.Kt |> clamp
    # println("\t- R $r")
    # println("\t- T $t")
    is_black(r) && is_black(t) && return

    is_specular = u_roughness ≈ 0 && v_roughness ≈ 0
    if is_specular && allow_multiple_lobes
        add!(si.bsdf, FresnelSpecular(r, t, 1f0, η, T))
        return
    end

    if g.remap_roughness
        u_roughness = roughness_to_α(u_roughness)
        v_roughness = roughness_to_α(v_roughness)
    end
    distribution = is_specular ? nothing : TrowbridgeReitzDistribution(
        u_roughness, v_roughness,
    )

    if !is_black(r)
        fresnel = FresnelDielectric(1f0, η)
        if is_specular
            # println("\t+ Adding SpecularReflection")
            add!(si.bsdf, SpecularReflection(r, fresnel))
        else
            # println("\t+ Adding MicrofacetReflection")
            add!(si.bsdf, MicrofacetReflection(r, distribution, fresnel, T))
        end
    end
    if !is_black(t)
        if is_specular
            # println("\t+ Adding SpecularTransmission")
            add!(si.bsdf, SpecularTransmission(t, 1f0, η, T))
        else
            # println("\t+ Adding MicrofacetTransmission")
            add!(si.bsdf, MicrofacetTransmission(t, distribution, 1f0, η, T))
        end
    end
end
