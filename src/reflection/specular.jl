struct SpecularReflection{S <: Spectrum, F <: Frensel} <: BxDF
    """
    Spectrum used to scale the reflected color.
    """
    r::S
    """
    Describes frensel properties.
    """
    frensel::F

    type::UInt8

    function SpecularReflection(r::S, frensel::F) where {S <: Spectrum, F <: Frensel}
        new{S, F}(r, frensel, BSDF_SPECULAR | BSDF_REFLECTION)
    end
end

"""
Return value of the distribution function for the given pair of directions.
For specular reflection, no scattering is returned, since
for arbitrary directions δ-funcion returns no scattering.
"""
function (s::SpecularReflection{S, F})(
    wo::Vec3f0, wi::Vec3f0,
) where {S <: Spectrum, F <: Frensel}
    S(0f0)
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
`sample` parameter isn't needed for the δ-distribution.
"""
function sample_f(
    s::SpecularReflection{S, F}, wo::Vec3f0, sample::Point2f0,
) where {S <: Spectrum, F <: Frensel}
    wi = Vec3f0(-wo[1], -wo[2], wo[3])
    pdf = 1f0
    wi, pdf, s.frensel(cos_θ(wi)) * s.r / abs(cos_θ(wi))
end

struct SpecularTransmission{S <: Spectrum, T <: TransportMode} <: BxDF
    t::S
    """
    Index of refraction above the surface.
    Side the surface normal lies in is "above".
    """
    η_a::Float32
    """
    Index of refraction below the surface.
    Side the surface normal lies in is "above".
    """
    η_b::Float32
    frensel::FrenselDielectric

    type::UInt8

    function SpecularTransmission{S, T}(
        t::S, η_a::Float32, η_b::Float32, frensel::FrenselDielectric,
    ) where {S <: Spectrum, T <: TransportMode}
        new{S, T}(t, η_a, η_b, frensel, BSDF_SPECULAR | BSDF_TRANSMISSION)
    end
end

"""
Return value of the distribution function for the given pair of directions.
For specular transmission, no scattering is returned, since
for arbitrary directions δ-funcion returns no scattering.
"""
function (s::SpecularTransmission{S, T})(
    wo::Vec3f0, wi::Vec3f0,
) where {S <: Spectrum, T <: TransportMode}
    S(0f0)
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
`sample` parameter isn't needed for the δ-distribution.
"""
function sample_f(
    s::SpecularTransmission{S, T}, wo::Vec3f0, sample::Point2f0,
)::Tuple{Vec3f0, Float32, S} where S <: Spectrum where T <: TransportMode
    # Figure out which η is incident and which is transmitted.
    entering = cos_θ(wo) > 0
    η_i = entering ? s.η_a : s.η_b
    η_t = entering ? s.η_b : s.η_a
    # Compute ray direction for specular transmission.
    valid, wi = refract(
        wo, face_forward(Normal3f0(0f0, 0f0, 1f0), wo), η_i / η_t,
    )
    !valid && return Vec3f0(0f0), 0f0, S(0f0) # Total internal reflection.
    pdf = 1f0

    cos_wi = wi |> cos_θ
    ft = s.t * (S(1f0) - s.frensel(cos_wi))
    # Account for non-symmetry with transmission to different medium.
    T isa Radiance && (ft *= (η_i ^ 2) / (η_t ^ 2))
    wi, pdf, ft / abs(cos_wi)
end

