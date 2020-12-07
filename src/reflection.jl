@enum BxDFTypes::UInt8 begin
    BSDF_NONE         = 0x0
    BSDF_REFLECTION   = 0x1
    BSDF_TRANSMISSION = 0b10
    BSDF_DIFFUSE      = 0b100
    BSDF_GLOSSY       = 0b1000
    BSDF_SPECULAR     = 0b10000
    BSDF_ALL          = 0b11111
end
abstract type BxDF end


"""
Compute Frensel reflection formula for dielectric materials
and unpolarized light.
Which describes the amount of light reflected from a surface.

- `cos_θi::Float32`: Cosine of the incident angle w.r.t. normal.
- `ηi::Float32`: index of refraction for the incident media.
- `ηt::Float32`: index of refraction for the transmitted media.
"""
function frensel_dielectric(cos_θi::Float32, ηi::Float32, ηt::Float32)
    cos_θi = clamp(cos_θi, -1f0, 1f0)
    if cos_θi > 0f0 # entering
        ηi, ηt = ηt, ηi
        cos_θi = cos_θi |> abs
    end
    # Compute cos_θt using Snell's law.
    sin_θi = max(0f0, 1f0 - cos_θi ^ 2) |> sqrt
    sin_θt = ηi / ηt * sin_θi
    sin_θt ≥ 1f0 && return 1f0 # Handle total internal reflection.
    cos_θt = sqrt(max(0f0, 1f0 - sin_θt ^ 2))

    r_parallel = ((ηt * cos_θi) - (ηi * cos_θt)) / ((ηt * cos_θi) + (ηi * cos_θt))
    r_perp = ((ηi * cos_θi) - (ηt * cos_θt)) / ((ηi * cos_θi) + (ηt * cos_θt))
    (r_parallel ^ 2 + r_perp ^ 2) / 2f0
end

"""
General Frensel reflection formula with complex index of refraction η^ = η + ik,
where some incident light is potentially absorbed by the material and turned into heat.
k - is referred to as the absorption coefficient.
"""
function frensel_conductor(cos_θi::Float32, ηi::S, ηt::S, k::S)::S where S <: Spectrum
    cos_θi = clamp(cos_θi, -1f0, 1f0)
    η = ηt / ηi
    ηk = k / ηi

    cos_θi2 = cos_θi * cos_θi
    sin_θi2 = 1f0 - cos_θi2
    η2 = η * η
    ηk2 = ηk * ηk

    t0 = η2 - ηk2 - sin_θi2
    a2_plus_b2 = sqrt(t0 * t0 + 4f0 * η2 * ηk2)
    t1 = a2_plus_b2 + cos_θi2
    a = sqrt(0.5f0 * (a2_plus_b2 + t0))
    t2 = 2f0 * cos_θi * a
    r_perp = (t1 - t2) / (t1 + t2)

    t3 = cos_θi2 * a2_plus_b2 + sin_θi2 * sin_θi2
    t4 = t2 * sin_θi2
    r_parallel = r_perp * (t3 - t4) / (t3 + t4)
    0.5f0 * (r_parallel + r_perp)
end

abstract type Frensel end
struct FrenselConductor{S <: Spectrum} <: Frensel
    ηi::S
    ηt::S
    k::S
end
struct FrenselDielectric <: Frensel
    ηi::Float32
    ηt::Float32
end
struct FrenselNoOp <: Frensel end
(f::FrenselConductor)(cos_θi::Float32) = frensel_conductor(cos_θi, f.ηi, f.ηt, f.k)
(f::FrenselDielectric)(cos_θi::Float32) = frensel_dielectric(cos_θi, f.ηi, f.ηt)
(f::FrenselNoOp)(::Float32) = RGBSpectrum(1f0)

struct SpecularReflection{S <: Spectrum, F <: Frensel} <: BxDF
    """
    Spectrum used to scale the reflected color.
    """
    r::S
    """
    Describes frensel properties.
    """
    frensel::F
end

"""
Return value of the distribution function for the given pair of directions.
For specular reflection, no scattering is returned, since
for arbitrary directions δ-funcion returns no scattering.
"""
function (s::SpecularReflection{S, F})(
    wo::Vec3f0, wi::Vec3f0,
)::S where S <: Spectrum where F <: Frensel
    S(0f0)
end

function Base.:&(::SpecularReflection, t::BxDFTypes)::Bool
    t & BSDF_SPECULAR || t & BSDF_REFLECTION
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
`sample` parameter isn't needed for the δ-distribution.
"""
function sample_f(
    s::SpecularReflection{S, F}, wo::Vec3f0, sample::Point2f0,
)::Tuple{Vec3f0, Float32, S} where S <: Spectrum where F <: Frensel
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
end

"""
Return value of the distribution function for the given pair of directions.
For specular transmission, no scattering is returned, since
for arbitrary directions δ-funcion returns no scattering.
"""
function (s::SpecularTransmission{S, T})(
    wo::Vec3f0, wi::Vec3f0,
)::S where S <: Spectrum where T <: TransportMode
    S(0f0)
end

function Base.:&(::SpecularTransmission, t::BxDFTypes)::Bool
    t & BSDF_SPECULAR || t & BSDF_TRANSMISSION
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

"""
Compute refracted direction `wt` given an incident direction `wi`,
surface normal `n` in the same hemisphere as `wi` and `η`, the ratio
of indices of refraction in the incident transmitted media respectively.

Returned boolean indicates whether a valid refracted ray was returned
or is it the case of total internal reflection.
"""
function refract(wi::Vec3f0, n::Normal3f0, η::Float32)::Tuple{Bool, Vec3f0}
    # Compute cosθt using Snell's law.
    cos_θi = n ⋅ wi
    sin2_θi = max(0f0, 1f0 - cos_θi ^ 2)
    sin2_θt = (η ^ 2) * sin2_θi
    # Handle total internal reflection for transmission.
    sin2_θt >= 1 && return false, Vec3f0(0f0)
    cos_θt = sqrt(1f0 - sin2_θt)
    wt = -η .* wi + (η * cos_θi - cos_θt) .* n
    true, wt
end


"""
Lambertian Reflection models a perfect diffuse surface
that scatters incident illumination equally in all directions.
"""
struct LambertianReflection{S <: Spectrum}
    """
    Reflectance spectrum, which is the fraction
    of incident light that is scattered.
    """
    r::S
end

function Base.:&(::LambertianReflection, t::BxDFTypes)::Bool
    t & BSDF_DIFFUSE || t & BSDF_REFLECTION
end

"""
Reflection distribution is constant and divides reflectance spectrum
equally over the hemisphere.
"""
function (l::LambertianReflection{S})(::Vec3f0, ::Vec3f0)::S where S <: Spectrum
    l.r * (1f0 / π)
end

"""
Directional-hemisphirical reflectance value is constant.
"""
function ρ(
    l::LambertianReflection{S}, ::Vec3f0, ::Int32, ::Vector{Point2f0},
)::S where S <: Spectrum
    l.r
end

"""
Hemispherical-hemisphirical reflectance value is constant.
"""
function ρ(
    l::LambertianReflection{S}, ::Vector{Point2f0}, ::Vector{Point2f0},
)::S where S <: Spectrum
    l.r
end


"""
Lambertian Transmission models perfect transmission.
"""
struct LambertianTransmission{S <: Spectrum}
    t::S
end

function Base.:&(::LambertianTransmission, t::BxDFTypes)::Bool
    t & BSDF_DIFFUSE || t & BSDF_TRANSMISSION
end

function (t::LambertianTransmission{S})(::Vec3f0, ::Vec3f0) where S <: Spectrum
    t.t * (1f0 / π)
end

function ρ(
    t::LambertianTransmission{S}, ::Vec3f0, ::Int32, ::Vector{Point2f0},
)::S where S <: Spectrum
    t.t
end

function ρ(
    t::LambertianTransmission{S}, ::Vector{Point2f0}, ::Vector{Point2f0},
)::S where S <: Spectrum
    t.t
end


"""
Describes rough surfaces by V-shaped microfacets described by a spherical
Gaussian distribution with parameter `σ` --- the standard deviation
of the microfacet angle.
"""
struct OrenNayar{S <: Spectrum} <: BxDF
    r::S
    a::Float32
    b::Float32

    function OrenNayar(r::S, σ::Float32) where S <: Spectrum
        σ = σ |> deg2rad
        σ2 = σ * σ
        a = 1f0 - (σ2 / (2f0 * (σ2 + 0.33f0)))
        b = 0.45f0 * σ2 / (σ2 + 0.09f0)
        new{S}(r, a, b)
    end
end

function (o::OrenNayar)(wo::Vec3f0, wi::Vec3f0)
    sin_θi = wi |> sin_θ
    sin_θo = wo |> sin_θ
    # Compute cosine term of Oren-Nayar model.
    max_cos = 0f0
    if sin_θi > 1f-4 && sin_θo > 1f-4
        sin_ϕi = wi |> sin_ϕ
        cos_ϕi = wi |> cos_ϕ
        sin_ϕo = wo |> sin_ϕ
        cos_ϕo = wo |> cos_ϕ
        max_cos = max(0f0, cos_ϕi * cos_ϕo + sin_ϕi * sin_ϕo)
    end
    # Compute sine & tangent terms of Oren-Nayar model.
    if abs(cos_θ(wi) > abs(cos_θ(wo)))
        sin_α = sin_θo
        tan_β = sin_θi / abs(cos_θ(wi))
    else
        sin_α = sin_θi
        tan_β = sin_θo / abs(cos_θ(wo))
    end
    o.r * (1f0 / π) * (o.a + o.b * max_cos * sin_α * tan_β)
end
