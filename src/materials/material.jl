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
        # @info "Adding Lambertian"
        add!(si.bsdf, LambertianReflection(r))
    else
        # @info "Adding OrenNayar"
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
    # @info "Adding Mirror"
    add!(si.bsdf, SpecularReflection(r, FrenselNoOp()))
end
