const TextureType = Union{Float32,S} where S<:Spectrum

# Texture wraps actual texture data arrays. For constant values, use raw values directly
# in material fields (materials should have loose type parameters).
struct Texture{ElType, N, T<:AbstractArray{ElType, N}}
    data::T
    constval::ElType
    isconst::Bool
end
Texture(data::AbstractArray{T,N}) where {T,N} = Texture{T,N,typeof(data)}(data, zero(T), false)
function ConstTexture(val::T) where {T}
    arr = Array{T,0}(undef)
    Texture{T,0,typeof(arr)}(arr, val, true)
end

Base.zero(::Type{RGBSpectrum}) = RGBSpectrum(0.0f0, 0.0f0, 0.0f0, 1.0f0)

# Sample texture data array with UV flip (standard texture coordinate convention)
@propagate_inbounds function sample_texture_data(data::AbstractArray{T,N}, uv::Point2f)::T where {T,N}
    uv_adj = Vec2f(1f0 - uv[2], uv[1])
    s = unsafe_trunc.(Int32, size(data))
    idx = map(x -> unsafe_trunc(Int32, x), Int32(1) .+ ((s .- Int32(1)) .* uv_adj))
    idx = clamp.(idx, Int32(1), s)
    return data[idx...]
end

# 0-dim arrays are scalar constants - just return the value, no UV sampling
@propagate_inbounds function sample_texture_data(data::AbstractArray{T,0}, ::Point2f)::T where T
    return data[]
end

function (c::Texture{T})(si::SurfaceInteraction)::T where {T<:TextureType}
    return sample_texture_data(c.data, si.uv)
end

# UV-only texture evaluation
@propagate_inbounds function evaluate_texture(tex::Texture{T}, uv::Point2f)::T where T
    tex.isconst && return tex.constval
    return sample_texture_data(tex.data, uv)
end

# Per-face vertex color texture: stores 3 colors per face for barycentric interpolation
struct VertexColorTexture{T}
    face_colors::T   # (3, n_faces) matrix of RGBSpectrum, becomes TextureRef via MultiTypeSet
    n_faces::Int32
end

# ============================================================================
# Auto-wrapping helpers: convert raw values to Texture
# ============================================================================

to_texture(t::Texture) = t
to_texture(v::RGBSpectrum) = ConstTexture(v)
to_texture(v::Float32) = ConstTexture(v)
to_texture(v::Real) = to_texture(Float32(v))
# For color tuples/vectors (use Tuple{Real,Real,Real} to handle mixed Int/Float)
to_texture(v::Tuple{Real,Real,Real}) = to_texture(RGBSpectrum(Float32(v[1]), Float32(v[2]), Float32(v[3])))
# Support Colors.jl RGB types (RGB, RGBA, etc.)
to_texture(c::Colorant) = to_texture(RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c))))
to_texture(c::AbstractMatrix{<: RGB}) = Texture(map(c-> RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c))), c))
# Identity passthrough for PiecewiseLinearSpectrum (not a texture, stored directly)
to_texture(s::PiecewiseLinearSpectrum) = s

