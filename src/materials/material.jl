struct MatterMaterial <: Material
    Kd::Texture
    σ::Texture
    # TODO bump map
end

"""
Compute scattering function.
"""
function (m::MatterMaterial)(
    si::SurfaceInteraction, allow_multiple_lobes::Bool, mode::TransportMode,
)
    # TODO perform bump mapping
    # TODO first implement
    #   + LambertianReflection
    #   + OrenNayar
    #   MicrofacetReflection
    #   TrowbridgeReitzDistribution

    # Evaluate textures and create BSDF.
    si.bsdf = BSDF(si)
    r = si |> m.Kd |> clamp
    is_black(r) && return

    σ = clamp(m.σ(si), 0, 90)
    if σ ≈ 0
        add!(si.bsdf, LambertianReflection(r))
    else
        add!(si.bsdf, OrenNayar(r, σ))
    end
end
