const TextureType = Union{Float32,S} where S<:Spectrum

struct Texture{ElType, N, T<:AbstractArray{ElType, N}}
    data::T
    const_value::ElType
    isconst::Bool
    function Texture{ElType, N, T}() where {ElType, N, T}
        new{ElType, N, T}()
    end
    function Texture(data::AbstractArray{T, N}, const_value::T, isconst::Bool) where {T, N}
        new{T, N, typeof(data)}(data, const_value, isconst)
    end
end
Base.zero(::Type{RGBSpectrum}) = RGBSpectrum(0.0f0, 0.0f0, 0.0f0)

Texture(data::AbstractArray{ElType, N}) where {ElType, N} = Texture(data, zero(ElType), false)
Texture(data::Eltype) where Eltype = Texture(Matrix{Eltype}(undef, 0, 0), data, true)
ConstantTexture(data::Eltype) where Eltype = Texture(data)
Texture() = Texture(0.0f0)

struct NoTexture end

function Base.convert(::Type{Texture{ElType,N,T}}, ::NoTexture) where {ElType,N,T}
    return Texture{ElType,N,T}()
end

function (c::Texture{T})(si::SurfaceInteraction)::T where {T<:TextureType}
    if c.isconst
        return c.const_value
    else
        uv = Vec2f(1.0-si.uv[2], si.uv[1])
        idx = round.(Int, 1 .+ ((size(c.data) .- 1) .* uv))
        idx = clamp.(idx, 1, size(c.data))
        return c.data[idx...]
    end
end
