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
    si::SurfaceInteraction, allow_multiple_lobes::Bool, mode::TransportMode,
)
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
    si::SurfaceInteraction, allow_multiple_lobes::Bool, mode::TransportMode,
)
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
    si::SurfaceInteraction, allow_multiple_lobes::Bool, mode::TransportMode,
)
    η = si |> g.index
    u_roughness = si |> g.u_roughness
    v_roughness = si |> g.v_roughness

    si.bsdf = BSDF(si, η)
    r = si |> g.Kr |> clamp
    t = si |> g.Kt |> clamp
    is_black(r) && is_black(t) && return

    is_specular = u_roughness ≈ 0 && v_roughness ≈ 0
    if is_specular && allow_multiple_lobes
        add!(si.bsdf, FresnelSpecular{mode}(r, t, 1f0, η))
        return
    end

    if g.remap_roughness
        u_roughness = roughness_to_α(u_roughness)
        v_roughness = roughness_to_α(v_roughness)
    end
    distribution = (
        is_specular
        ? nothing
        : TrowbridgeReitzDistribution(u_roughness, v_roughness)
    )

    if !is_black(r)
        fresnel = FresnelDielectric(1f0, η)
        if is_specular
            add!(si.bsdf, SpecularReflection(r, fresnel))
        else
            add!(si.bsdf, MicrofacetReflection{mode}(r, distribution, fresnel))
        end
    end
    if !is_black(t)
        if is_specular
            add!(si.bsdf, SpecularTransmission{mode}(t, 1f0, η))
        else
            add!(si.bsdf, MicrofacetTransmission{mode}(t, distribution, 1f0, η))
        end
    end

    """
    TODO
    + FresnelSpecular
    - MicrofacetReflection
    - MicrofacetTransmission
    """
end
