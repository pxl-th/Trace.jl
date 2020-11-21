const NSpectralSamples = 60
const NRGB2SpectralSamples = 32
const NCIESamples = 471
const λ_start = 400f0
const λ_end = 700f0

@inline function XYZ_to_RGB(xyz::Point3f0)
    Point3f0(
        3.240479f0 * xyz[1] - 1.537150f0 * xyz[2] - 0.498535f0 * xyz[3],
        -0.969256f0 * xyz[1] + 1.875991f0 * xyz[2] + 0.041556f0 * xyz[3],
        0.055648f0 * xyz[1] - 0.204043f0 * xyz[2] + 1.057311f0 * xyz[3],
    )
end
@inline function RGB_to_XYZ(rgb::Point3f0)
    Point3f0(
        0.412453f0 * rgb[1] + 0.357580f0 * rgb[2] + 0.180423f0 * rgb[3],
        0.212671f0 * rgb[1] + 0.715160f0 * rgb[2] + 0.072169f0 * rgb[3],
        0.019334f0 * rgb[1] + 0.119193f0 * rgb[2] + 0.950227f0 * rgb[3],
    )
end

include("spectrum_data.jl")

abstract type Spectrum end

Base.:(==)(c::C, i) where C <: Spectrum = all(c.c .== i)
Base.:(==)(c1::C, c2::C) where C <: Spectrum = all(c1.c .== c2.c)
Base.:+(c1::C, c2::C) where C <: Spectrum = C(c1.c .+ c2.c)
Base.:+(c1::C, c::Float32) where C <: Spectrum = C(c1.c .+ c)
Base.:-(c::C) where C <: Spectrum = -c.c |> C(-c.c)
Base.:-(c1::C, c::Float32) where C <: Spectrum = C(c1.c .- c)
Base.:-(c1::C, c2::C) where C <: Spectrum = C(c1.c .- c2.c)
Base.:*(c1::C, c2::C) where C <: Spectrum = C(c1.c .* c2.c)
Base.:*(c1::C, f::Float32) where C <: Spectrum = C(c1.c .* f)
Base.:*(f::Float32, c1::C) where C <: Spectrum = C(c1.c .* f)
Base.:/(c1::C, c2::C) where C <: Spectrum = C(c1.c ./ c2.c)
Base.:/(c1::C, f::Float32) where C <: Spectrum = C(c1.c ./ f)
Base.sqrt(c::C) where C <: Spectrum = C(c.c .|> sqrt)
Base.:^(c::C, e::Float32) where C <: Spectrum = C(c.c .^ e)
Base.exp(c::C, e::Float32) where C <: Spectrum = C(c.c .|> exp)
lerp(c1::C, c2::C, t::Float32) where C <: Spectrum = (1f0 - t) * c1 + t * c2

Base.getindex(c::C, i) where C <: Spectrum = c.c[i]
function Base.clamp(c::C, low::Float32 = 0f0, high::Float32 = Inf32) where C <: Spectrum
    C(clamp.(c.c, low, high))
end
function Base.isnan(c::C) where C <: Spectrum
    for v in c.c
        isnan(v) && return true
    end
    false
end

is_black(c::C) where C <: Spectrum = iszero(c.c)

struct CoefficientSpectrum <: Spectrum
    c::Vector{Float32}
end

function CoefficientSpectrum(v::Float32 = 0f0)
    CoefficientSpectrum(fill(v, NSpectralSamples))
end

struct SampledSpectrum <: Spectrum
    c::Vector{Float32}
end

function SampledSpectrum(v::Float32 = 0f0)
    SampledSpectrum(fill(v, NSpectralSamples))
end

const CIE_Y_integral = 106.856895f0
const X = SampledSpectrum()
const Y = SampledSpectrum()
const Z = SampledSpectrum()
const rgbRefl2SpectWhite = SampledSpectrum()
const rgbRefl2SpectCyan = SampledSpectrum()
const rgbRefl2SpectMagenta = SampledSpectrum()
const rgbRefl2SpectYellow = SampledSpectrum()
const rgbRefl2SpectRed = SampledSpectrum()
const rgbRefl2SpectGreen = SampledSpectrum()
const rgbRefl2SpectBlue = SampledSpectrum()
const rgbIllum2SpectWhite = SampledSpectrum()
const rgbIllum2SpectCyan = SampledSpectrum()
const rgbIllum2SpectMagenta = SampledSpectrum()
const rgbIllum2SpectYellow = SampledSpectrum()
const rgbIllum2SpectRed = SampledSpectrum()
const rgbIllum2SpectGreen = SampledSpectrum()
const rgbIllum2SpectBlue = SampledSpectrum()

function from_sampled(::Type{SampledSpectrum}, λ::Vector{Float32}, v::Vector{Float32})
    !issorted(λ) && (λ, v = sort_sampled(λ, v))
    r = SampledSpectrum()
    n = 1f0 / Float32(NSpectralSamples)
    @inbounds for i in 1:NSpectralSamples
        λ0 = lerp(λ_start, λ_end, i * n)
        λ1 = lerp(λ_start, λ_end, (i + 1f0) * n)
        r.c[i] = average_spectrum_samples(λ, v, λ0, λ1)
    end
end

function sort_sampled(λ::Vector{Float32}, v::Vector{Float32})
    perm = λ |> sortperm
    λ[perm], v[perm]
end

function average_spectrum_samples(
    λ::Vector{Float32}, v::Vector{Float32}, λ0::Float32, λ1::Float32,
)
    # Handle out-of-bounds cases.
    λ1 <= λ[begin] && return v[begin]
    λ0 >= λ[end] && return v[end]
    length(λ) == 1 && return v[begin]

    Σ = 0f0
    # Add contributions of constant segments before/after samples.
    λ0 < λ[1] && (Σ += v[1] * (λ[1] - λ0))
    λ1 > λ[end] && (Σ += v[end] * (λ1 - λ[end]))
    # Advance to first relevant segment.
    i = 1
    while λ0 > λ[i]
        i += 1
    end
    @assert i <= length(λ)
    # Loop over segment wavelength sample segments & add contribution.
    interpolate = (w::Float32, i::Int64) -> lerp(v[i], v[i + 1], (w - λ[i]) / (λ[i + 1] - λ[i]))
    n = length(λ) + 1
    while i + 1 < n && λ1 >= λ[i]
        sλ0, sλ1 = max(λ0, λ[i]), min(λ1, λ[i + 1])
        Σ += 0.5f0 * (interpolate(sλ0, i) + interpolate(sλ1, i)) * (sλ1 - sλ0)
        i += 1
    end
    Σ / (λ1 - λ0)
end

function init()
    n = 1f0 / Float32(NSpectralSamples)
    @inbounds for i in 1:NSpectralSamples
        w0 = lerp(λ_start, λ_end, i * n)
        w1 = lerp(λ_start, λ_end, (i + 1f0) * n)
        X.c[i] = average_spectrum_samples(CIE_λ, CIE_X, w0, w1)
        Y.c[i] = average_spectrum_samples(CIE_λ, CIE_Y, w0, w1)
        Z.c[i] = average_spectrum_samples(CIE_λ, CIE_Z, w0, w1)

        rgbRefl2SpectWhite.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectWhite, w0, w1)
        rgbRefl2SpectCyan.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectCyan, w0, w1)
        rgbRefl2SpectMagenta.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectMagenta, w0, w1)
        rgbRefl2SpectYellow.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectYellow, w0, w1)
        rgbRefl2SpectRed.c[i] = average_spectrum_samples( RGB2SpectLambda, RGBRefl2SpectRed, w0, w1)
        rgbRefl2SpectGreen.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectGreen, w0, w1)
        rgbRefl2SpectBlue.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectBlue, w0, w1)

        rgbIllum2SpectWhite.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectWhite, w0, w1)
        rgbIllum2SpectCyan.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectCyan, w0, w1)
        rgbIllum2SpectMagenta.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectMagenta, w0, w1)
        rgbIllum2SpectYellow.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectYellow, w0, w1)
        rgbIllum2SpectRed.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectRed, w0, w1)
        rgbIllum2SpectGreen.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectGreen, w0, w1)
        rgbIllum2SpectBlue.c[i] = average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectBlue, w0, w1)
    end
end

function to_XYZ(c::SampledSpectrum)
    x, y, z = 0f0, 0f0, 0f0
    @inbounds for i in 1:NSpectralSamples
        x += X.c[i] * c.c[i]
        y += Y.c[i] * c.c[i]
        z += Z.c[i] * c.c[i]
    end
    scale = (λ_end - λ_start) / (CIE_Y_integral * Float32(NSpectralSamples))
    Point3f0(x * scale, y * scale, z * scale)
end

function to_y(c::SampledSpectrum)
    y = 0f0
    @inbounds for i in 1:NSpectralSamples
        y += Y.c[i] * c.c[i]
    end
    scale = (λ_end - λ_start) / Float32(NSpectralSamples)
    y * scale
end

@inline to_RGB(c::SampledSpectrum) = c |> to_XYZ |> XYZ_to_RGB

@enum SpectrumType Reflectance Illuminant

function from_RGB(::Type{SampledSpectrum}, rgb::Point3f0, type::SpectrumType = Reflectance)
    r = SampledSpectrum()
    if type == Reflectance
        if rgb[1] <= rgb[2] && rgb[1] <= rgb[3]
            r += rgb[1] * rgbRefl2SpectWhite
            if rgb[2] <= rgb[3]
                r += (rgb[2] - rgb[1]) * rgbRefl2SpectCyan
                r += (rgb[3] - rgb[2]) * rgbRefl2SpectBlue
            else
                r += (rgb[3] - rgb[1]) * rgbRefl2SpectCyan
                r += (rgb[2] - rgb[3]) * rgbRefl2SpectGreen
            end
        elseif rgb[2] <= rgb[1] && rgb[2] <= 3
            r += rgb[2] * rgbRefl2SpectWhite
            if rgb[1] <= rgb[3]
                r += (rgb[1] - rgb[2]) * rgbRefl2SpectMagenta
                r += (rgb[3] - rgb[1]) * rgbRefl2SpectBlue
            else
                r += (rgb[3] - rgb[2]) * rgbRefl2SpectMagenta
                r += (rgb[1] - rgb[3]) * rgbRefl2SpectRed
            end
        else
            r += rgb[3] * rgbRefl2SpectWhite
            if rgb[1] <= rgb[2]
                r += (rgb[1] - rgb[3]) * rgbRefl2SpectYellow
                r += (rgb[2] - rgb[1]) * rgbRefl2SpectGreen
            else
                r += (rgb[2] - rgb[3]) * rgbRefl2SpectYellow
                r += (rgb[1] - rgb[2]) * rgbRefl2SpectRed
            end
        end
    else
        if rgb[1] <= rgb[2] && rgb[1] <= rgb[3]
            r += rgb[1] * rgbIllum2SpectWhite
            if rgb[2] <= rgb[3]
                r += (rgb[2] - rgb[1]) * rgbIllum2SpectCyan
                r += (rgb[3] - rgb[2]) * rgbIllum2SpectBlue
            else
                r += (rgb[3] - rgb[1]) * rgbIllum2SpectCyan
                r += (rgb[2] - rgb[3]) * rgbIllum2SpectGreen
            end
        elseif rgb[2] <= rgb[1] && rgb[2] <= 3
            r += rgb[2] * rgbIllum2SpectWhite
            if rgb[1] <= rgb[3]
                r += (rgb[1] - rgb[2]) * rgbIllum2SpectMagenta
                r += (rgb[3] - rgb[1]) * rgbIllum2SpectBlue
            else
                r += (rgb[3] - rgb[2]) * rgbIllum2SpectMagenta
                r += (rgb[1] - rgb[3]) * rgbIllum2SpectRed
            end
        else
            r += rgb[3] * rgbIllum2SpectWhite
            if rgb[1] <= rgb[2]
                r += (rgb[1] - rgb[3]) * rgbIllum2SpectYellow
                r += (rgb[2] - rgb[1]) * rgbIllum2SpectGreen
            else
                r += (rgb[2] - rgb[3]) * rgbIllum2SpectYellow
                r += (rgb[1] - rgb[2]) * rgbIllum2SpectRed
            end
        end
    end
    r |> clamp
end

@inline function from_XYZ(
    ::Type{SampledSpectrum}, xyz::Point3f0, type::SpectrumType = Reflectance,
)
    from_RGB(SampledSpectrum, xyz |> XYZ_to_RGB, type)
end

struct RGBSpectrum <: Spectrum
    c::Vector{Float32}
end

RGBSpectrum(v::Float32 = 0f0) = RGBSpectrum(fill(v, 3))

@inline from_RGB(::Type{RGBSpectrum}, rgb::Point3f0, ::SpectrumType = Reflectance) = rgb |> RGBSpectrum
@inline to_RGB(s::RGBSpectrum) = Point3f0(s.c)
@inline to_XYZ(s::RGBSpectrum) = s |> to_RGB |> RGB_to_XYZ
@inline function from_XYZ(
    ::Type{RGBSpectrum}, xyz::Point3f0, type::SpectrumType = Reflectance,
)
    xyz |> XYZ_to_RGB |> RGBSpectrum
end

function from_sampled(::Type{RGBSpectrum}, λ::Vector{Float32}, v::Vector{Float32})
    !issorted(λ) && (λ, v = sort_sampled(λ, v))
    xyz = Float32[0f0, 0f0, 0f0]
    @inbounds for i in 1:NCIESamples
        val = interpolate_spectrum_samples(λ, v, CIE_λ[i])
        xyz[1] += val * CIE_X[i]
        xyz[2] += val * CIE_Y[i]
        xyz[3] += val * CIE_Z[i]
    end

    scale = (CIE_λ[end] - CIE_λ[1]) / (CIE_Y_integral * NCIESamples)
    xyz .*= scale
    from_XYZ(RGBSpectrum, xyz |> Point3f0)
end

function interpolate_spectrum_samples(
    λ::Vector{Float32}, v::Vector{Float32}, l::Float32,
)
    l <= λ[begin] && return λ[begin]
    l >= λ[end] && return λ[end]
    offset = find_interval(λ |> length, i::Int64 -> (λ[i] <= l))
    t = (l - λ[offset]) / (λ[offset + 1] - λ[offset])
    lerp(v[offset], v[offset + 1], t)
end

init()
