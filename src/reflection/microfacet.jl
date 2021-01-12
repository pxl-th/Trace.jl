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


abstract type MicrofacetDistribution <: BxDF end

"""
Microfacet distribution function based on Gaussian distribution of
microfacet slopes.
Distribution has higher tails, it falls off to zero more slowly for
directions far from the surface normal.
"""
struct TrowbridgeReitzDistribution <: MicrofacetDistribution
    α_x::Float32
    α_y::Float32
    sample_visible_area::Bool

    function TrowbridgeReitzDistribution(
        sample_visible_area::Bool, α_x::Float32, α_y::Float32,
    )
        new(max(1f-3, α_x), max(1f-3, α_y), sample_visible_area)
    end
end

function λ(trd::TrowbridgeReitzDistribution, w::Vec3f0)::Float32
    θ = w |> tan_θ |> abs
    isinf(θ) && return 0f0

    α = sqrt(cos_ϕ(w) ^ 2 * trd.α_x ^ 2 + sin_ϕ(w) ^ 2 * trd.α_y ^ 2)
    α²tanθ = (α * θ) ^ 2
    (-1f0 + sqrt(1f0 + α²tanθ)) / 2f0
end

"""
Map [0, 1] scalar to BRDF's roughness, where values close to zero
correspond to near-perfect specular reflection, rather than by specifying
α values directly.
"""
@inline function roughness_to_α(roughness::Float32)::Float32
    roughness = max(1f-3, roughness)
    x = log(roughness)
    1.62142f0 + 0.819955f0 * x + 0.1734f0 * x ^ 2 +
        0.0171201f0 * x ^ 3 + 0.000640711f * x ^ 4
end

"""
- G1
- G
- D
- PDF
- Sample_wh
"""
