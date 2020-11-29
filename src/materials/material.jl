abstract type Material end

struct MatterMaterial
    Kd::Texture
    σ::Texture
    # TODO bump map
end

function (m::MatterMaterial)(
    si::SurfaceInteraction, mode::TransportMode, allow_multiple_lobes::Bool,
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
