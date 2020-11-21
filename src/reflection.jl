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
struct FrenselNoOp end
(f::FrenselConductor)(cos_θi::Float32) = frensel_conductor(cos_θi, f.ηi, f.ηt, f.k)
(f::FrenselDielectric)(cos_θi::Float32) = frensel_dielectric(cos_θi, f.ηi, f.ηt)
(f::FrenselNoOp)(::Float32) = RGBSpectrum(1f0)


abstract type BxDF end
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
function f(
    s::SpecularReflection{S, F}, wo::Vec3f0, wi::Vec3f0,
)::S where S <: Spectrum where F <: Frensel
    S(0f0)
end

"""
Since normal is (0, 0, 1), cos_θ between n & w is (0, 0, 1) ⋅ w = w.z.
"""
@inline cos_θ(w::Vec3f0) = w[3]

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

"""
Reflect `wo` about `n`.
"""
@inline reflect(wo::Vec3f0, n::Vec3f0) = -wo + 2f0 * (wo ⋅ n) * n

const Radiance = Val{:Radiance}
const Importance = Val{:Importance}
const TransportMode = Union{Radiance, Importance}

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
function f(
    s::SpecularTransmission{S, T}, wo::Vec3f0, wi::Vec3f0,
)::S where S <: Spectrum where T <: TransportMode
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