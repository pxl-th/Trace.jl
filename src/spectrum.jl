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

Base.:(==)(c::C, i) where C <: Spectrum = all(c.c .== i)
Base.:(==)(c1::C, c2::C) where C <: Spectrum = all(c1.c .== c2.c)
Base.:+(c1::C, c2::C) where C <: Spectrum = C(c1.c .+ c2.c)
Base.:+(c1::C, c) where C <: Spectrum = C(c1.c .+ c)
Base.:-(c::C) where C <: Spectrum = -c.c |> C(-c.c)
Base.:-(c1::C, c::Number) where C <: Spectrum = C(c1.c .- c)
Base.:-(c1::C, c2::C) where C <: Spectrum = C(c1.c .- c2.c)
Base.:*(c1::C, c2::C) where C <: Spectrum = C(c1.c .* c2.c)
Base.:*(c1::C, f::Number) where C <: Spectrum = C(c1.c .* f)
Base.:*(f::Number, c1::C) where C <: Spectrum = C(c1.c .* f)
Base.:/(c1::C, c2::C) where C <: Spectrum = C(c1.c ./ c2.c)
Base.:/(c1::C, f::Number) where C <: Spectrum = C(c1.c ./ f)
Base.sqrt(c::C) where C <: Spectrum = C(c.c .|> sqrt)
Base.:^(c::C, e::Number) where C <: Spectrum = C(c.c .^ e)
Base.exp(c::C, e::Float32) where C <: Spectrum = C(c.c .|> exp)
lerp(c1::C, c2::C, t::Float32) where C <: Spectrum = (1f0 - t) * c1 + t * c2

Base.getindex(c::C, i) where C <: Spectrum = c.c[i]

function Base.clamp(
    c::C, low::Float32 = 0f0, high::Float32 = Inf32,
) where C <: Spectrum
    C(clamp.(c.c, low, high))
end

function Base.isnan(c::C) where C <: Spectrum
    for v in c.c
        isnan(v) && return true
    end
    false
end

function Base.isinf(c::C) where C <: Spectrum
    for v in c.c
        isinf(v) && return true
    end
    false
end

is_black(c::C) where C <: Spectrum = c.c |> iszero

struct RGBSpectrum <: Spectrum
    c::Point3f0
end
RGBSpectrum(v::Float32 = 0f0) = RGBSpectrum(Point3f0(v))
RGBSpectrum(r, g, b) = RGBSpectrum(Point3f0(r, g, b))

to_XYZ(s::RGBSpectrum) = s.c |> RGB_to_XYZ
@inbounds function to_Y(s::RGBSpectrum)::Float32
    0.212671f0 * s.c[1] + 0.715160f0 * s.c[2] + 0.072169f0 * s.c[3]
end
function from_XYZ(::Type{RGBSpectrum}, xyz::Point3f0)
    xyz |> XYZ_to_RGB |> RGBSpectrum
end
