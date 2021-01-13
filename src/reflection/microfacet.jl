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

    α = √(cos_ϕ(w) ^ 2 * trd.α_x ^ 2 + sin_ϕ(w) ^ 2 * trd.α_y ^ 2)
    α²tanθ² = (α * θ) ^ 2
    (-1f0 + √(1f0 + α²tanθ²)) / 2f0
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
        0.0171201f0 * x ^ 3 + 0.000640711f0 * x ^ 4
end

@inline function G1(m::MicrofacetDistribution, w::Vec3f0)::Float32
    1f0 / (1f0 + λ(m, w))
end

@inline function G(m::MicrofacetDistribution, wo::Vec3f0, wi::Vec3f0)::Float32
    1f0 / (1f0 + λ(m, wo) + λ(m, wi))
end

"""
Distribution function, which gives the differential area of microfacets
with the surface normal `w`.
"""
function D(trd::TrowbridgeReitzDistribution, w::Vec3f0)::Float32
    tan_θ² = tan_θ(w) ^ 2
    isinf(tan_θ²) && return 0f0

    cos_θ⁴ = cos_θ(w) ^ 4
    e = (cos_ϕ(w) ^ 2 / (trd.α_x ^ 2) + sin_ϕ(w) ^ 2 / (trd.α_y ^ 2)) * tan_θ²
    1f0 / (π * trd.α_x * trd.α_y * cos_θ⁴ * (1f0 + e) ^ 2)
end

function Pdf(m::MicrofacetDistribution, wo::Vec3f0, wh::Vec3f0)::Float32
    !m.sample_visible_area && return D(m, wh) * abs(cos_θ(wh))
    D(m, wh) * G1(m, wo) * abs(wo ⋅ wh) / abs(cos_θ(wo))
end

function _trowbridge_reitz_sample(
    cosθ::Float32, u1::Float32, u2::Float32,
)::Tuple{Float32, Float32}
    if cosθ > 0.9999f0 # Special case -- normal incidence.
        r = √(u1 / (1f0 - u1))
        ϕ = 6.28318530718 * u2
        return r * cos(ϕ), r * sin(ϕ)
    end

    sinθ = √(max(0f0, 1f0 - cosθ ^ 2))
    tanθ = sinθ / cosθ
    a = 1f0 / tanθ
    g1 = 2f0 / (1f0 + √(1f0 + 1f0 / (a ^ 2)))

    # Sample slope x.
    a = 2f0 * u1 / g1 - 1f0
    tmp = 1f0 / (a ^ 2 - 1f0)
    tmp > 1f10 && (tmp = 1f10;)
    b = tanθ
    b² = b ^ 2
    d = √(max(0f0, b² * tmp ^ 2 - (a ^ 2 - b²) * tmp))
    slope_x1, slope_x2 = b * tmp - d, b * tmp + d
    slope_x = (a < 0 || slope_x2 > 1f0 / tanθ) ? slope_x1 : slope_x2

    # Sample slope y.
    if u2 > 0.5f0
        s = 1f0
        u2 = 2f0 * (u2 - 0.5f0)
    else
        s = -1f0
        u2 = 2f0 * (0.5f0 - u2)
    end
    z = (
        (u2 * (u2 * (u2 * 0.27385f0 - 0.73369f0) + 0.46341f0))
        / (u2 * (u2 * (u2 * 0.093073f0 + 0.309420f0) - 1f0) + 0.597999f0)
    )
    slope_y = s * z * √(1f0 + slope_x ^ 2)

    @assert !isinf(slope_y) && !isnan(slope_y)
    slope_x, slope_y
end

function trowbridge_reitz_sample(
    wi::Vec3f0, α_x::Float32, α_y::Float32, u1::Float32, u2::Float32,
)::Vec3f0
    # Stretch wi.
    wi_stretch = Vec3f0(wi[1] * α_x, wi[2] * α_y, wi[3]) |> normalize
    slope_x, slope_y = _trowbridge_reitz_sample(cos_θ(wi_stretch), u1, u2)
    # Rotate.
    c, s = cos_ϕ(wi_stretch), sin_ϕ(wi_stretch)
    tmp = c * slope_x - s * slope_y
    slope_y = s * slope_x + c * slope_y
    slope_x = tmp
    # Unstretch.
    slope_x *= α_x
    slope_y *= α_y
    # Compute normal.
    Vec3f0(-slope_x, -slope_y, 1f0) |> normalize
end

function sample_wh(
    trd::TrowbridgeReitzDistribution, wo::Vec3f0, u::Point2f0,
)::Vec3f0
    if trd.sample_visible_area
        flip = wo[3] < 0f0
        wh = trowbridge_reitz_sample(
            flip ? -wo : wo, trd.α_x, trd.α_y, u[1], u[2],
        )
        return flip ? -wh : wh
    end
    cosθ = 0f0
    ϕ = 2f0 * π * u[2]
    if trd.α_x ≈ trd.α_y
        tanθ² = trd.α_x ^ 2 * u[1] / (1f0 - u[1])
        cosθ = 1f0 / √(1f0 + tanθ²)
    else
        ϕ = atan(trd.α_y / trd.α_x * tan(2f0 * π * u[2] + 0.5f0 * π))
        u[2] > 0.5f0 && (ϕ += π)
        sinϕ, cosϕ = sin(ϕ), cos(ϕ)
        α_x², α_y² = trd.α_x ^ 2, trd.α_y ^ 2
        α² = 1f0 / (cosϕ ^ 2 / α_x² + sinϕ ^ 2 / α_y²)
        tanθ² = α² * u[1] / (1f0 - u[1])
        cosθ = 1f0 / √(1f0 + tanθ²)
    end

    sinθ = √(max(0f0, 1f0 - cosθ ^ 2))
    wh = spherical_direction(sinθ, cosθ, ϕ)
    same_hemisphere(wo, wh) ? wh : -wh
end
