"""
Compute Frensel reflection formula for dielectric materials
and unpolarized light.

ηi & ηt are the indices of refraction for the incident and transmitted media.
"""
function frensel_dielectric(cos_θi::Float32, ηi::Float32, ηt::Float32)
    cos_θi = clamp(cos_θi, -1f0, 1f0)
    if cos_θi > 0f0 # entering
        ηi, ηt = ηt, ηi
        cos_θi = cos_θi |> abs
    end
    # Compute cos_θt using Snell's law.
    sin_θi = sqrt(max(0f0, 1f0 - cos_θi ^ 2))
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
function frensel_conductor(cos_θi::Float32, ηi::S, ηt::S, k::S) where S <: Spectrum
    cos_θi = clamp(cos_θi, -1f0, 1f0)
    η = ηt / ηi
    ηk = k / ηi

    cos_θi2 = cos_θi * cos_θi
    sin_θi2 = 1f0 - cos_θi2
    η2 = η * η
    ηk2 = ηk * ηk

    t0 = η2 - ηk2 - sin_θi2
    a2_plus_b2 = sqrt(t0 * t0 + 4 * η2 * ηk2)
    t1 = a2_plus_b2 + cos_θi2
    a = sqrt(0.5f0 * (a2_plus_b2 + t0))
    t2 = 2f0 * cos_θi * a
    r_perp = (t1 - t2) / (t1 + t2)

    t3 = cos_θi2 * a2_plus_b2 + sin_θi2 * sin_θi2
    t4 = t2 * sin_θi2
    r_parallel = r_perp * (t3 - t4) / (t3 + t4)
    0.5f0 * (r_parallel + r_perp)
end

struct FrenselConductor{S}
    ηi::S
    ηt::S
    k::S
end
struct FrenselDielectric
    ηi::Float32
    ηt::Float32
end
struct FrenselNoOp end
(f::FrenselConductor)(cos_θi::Float32) = frensel_conductor(cos_θi, f.ηi, f.ηt, f.k)
(f::FrenselDielectric)(cos_θi::Float32) = frensel_dielectric(cos_θi, f.ηi, f.ηt)
(f::FrenselNoOp)(::Float32) = RGBSpectrum(1f0)
