abstract type Filter end

struct LanczosSincFilter <: Filter
    radius::Point2f0
    τ::Float32
end

function (f::LanczosSincFilter)(p::Point2f0)::Float32
    windowed_sinc(p[1], f.radius[1], f.τ) * windowed_sinc(p[2], f.radius[2], f.τ)
end

@inline function sinc(x::Float32)::Float32
    x = x |> abs
    x < 1f-5 && return 1f0
    x *= π |> Float32
    sin(x) / x
end

function windowed_sinc(x::Float32, r::Float32, τ::Float32)::Float32
    x = x |> abs
    x > r && return 0f0
    sinc(x) * sinc(x / τ)
end
