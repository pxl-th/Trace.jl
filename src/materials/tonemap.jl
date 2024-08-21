using Colors, Statistics

luminosity(c::RGB{T}) where {T} = (max(c.r, c.g, c.b) + min(c.r, c.g, c.b)) / 2.0

function lum_max(rgb_m)
    lum_max = 0.0
    for pix in rgb_m
        (lum_max > luminosity(pix)) || (lum_max = luminosity(pix))
    end
    lum_max
end

function avg_lum(rgb_m, δ::Number=1e-10)
    cumsum = 0.0
    for pix in rgb_m
        cumsum += log10(δ + luminosity(pix))
    end
    return 10^(cumsum / (prod(size(rgb_m))))
end

function normalize_image(
        rgb_m,
        a::Float64=0.18,
        lum::Union{Number,Nothing}=nothing,
        δ::Number=1e-10
    )

    (isnothing(lum) || lum ≈ 0.0) && (lum = avg_lum(rgb_m, δ))
    return rgb_m .* a .* (1.0 / lum)
end

function clamp_image(img::AbstractMatrix{T}) where {T}
    return map(img) do col
        return T(clamp(col.r, 0, 1), clamp(col.g, 0, 1), clamp(col.b, 0, 1))
    end
end

function γ_correction(img::AbstractMatrix{T}, γ::Float64=1.0, k::Float64=1.0) where T
    return map(img) do c
        return T(
            floor(255 * c.r^(1 / γ)),
            floor(255 * c.g^(1 / γ)),
            floor(255 * c.b^(1 / γ))
        )
    end
end

function tone_mapping(img;
        a::Float64=0.18,
        γ::Float64=1.0,
        lum::Union{Number,Nothing}=nothing
    )
    img = normalize_image(img, a, lum)
    img = clamp_image(img)
    return γ_correction(img, γ)
end

tone_mapping(img)
