"""
Describes rough surfaces by V-shaped microfacets described by a spherical
Gaussian distribution with parameter `σ` --- the standard deviation
of the microfacet angle.
"""
struct OrenNayar{S <: Spectrum} <: BxDF
    r::S
    a::Float32
    b::Float32
    type::UInt8

    function OrenNayar(r::S, σ::Float32) where S <: Spectrum
        σ = σ |> deg2rad
        σ2 = σ * σ
        a = 1f0 - (σ2 / (2f0 * (σ2 + 0.33f0)))
        b = 0.45f0 * σ2 / (σ2 + 0.09f0)
        new{S}(r, a, b, BSDF_DIFFUSE | BSDF_REFLECTION)
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
