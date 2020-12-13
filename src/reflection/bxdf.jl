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

@inline Base.:&(b::BxDFTypes, v::UInt8)::UInt8 = UInt8(b) & v
@inline Base.:&(v::UInt8, b::BxDFTypes)::UInt8 = UInt8(b) & v
function Base.:&(b::B, type::Union{UInt8, BxDFTypes})::Bool where B <: BxDF
    (b.type & type) != 0
end

@inline same_hemisphere(w::Vec3f0, wp::Vec3f0)::Bool = w[3] * wp[3] > 0
@inline same_hemisphere(w::Vec3f0, wp::Normal3f0)::Bool = w[3] * wp[3] > 0

"""
Compute PDF value for the given directions.
In comparison, `sample_f` computes PDF value for the incident directions *it*
chooses given the outgoing direction, while this returns a value of PDF
for the given pair of directions.
"""
@inline function compute_pdf(b::B, wo::Vec3f0, wi::Vec3f0)::Float32 where B <: BxDF
    same_hemisphere(wo, wi) ? abs(cos_θ(wi)) * (1f0 / π) : 0
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.

**Note** all BxDFs that implement this method,
have to implement `compute_pdf` as well.
"""
function sample_f(
    b::B, wo::Vec3f0, sample::Point2f0,
)::Tuple{Vec3f0, Float32, S} where {S <: Spectrum, B <: BxDF}
    wi = sample |> cosine_sample_hemisphere
    # Flipping the direction if necessary.
    wo[3] < 0 && (wi[3] *= -1)
    pdf = compute_pdf(b, wo, wi)
    wi, pdf, b(wo, wi)
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
