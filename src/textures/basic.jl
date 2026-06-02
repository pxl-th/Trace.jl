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

# Sample texture data array with bilinear interpolation. pbrt-v4 uses
# bilinear (and mipmap-trilinear) by default for imagemap textures. Without
# at least bilinear the BumpMap function sees `h(u+du) == h(u)` whenever
# du is sub-texel, which left Crown's gold dome with coherent mirror
# reflections because the bump gradient came out as exactly zero.
#
# UV convention matches the prior nearest-neighbour version (dim 1 is
# 1-v, dim 2 is u, repeat wrap via `frac`).
@propagate_inbounds function sample_texture_data(data::AbstractArray{T,N}, uv::Point2f)::T where {T,N}
    u_wrapped = uv[1] - floor(uv[1])
    v_wrapped = uv[2] - floor(uv[2])
    s = size(data)
    # Convert wrapped UV into fractional texel-space coordinates. The 0.5
    # offset puts (0,0) at the centre of texel (1,1) so the bilinear
    # weights cleanly span the four surrounding texels.
    fx = (1f0 - v_wrapped) * Float32(s[1]) - 0.5f0
    fy = u_wrapped * Float32(s[2]) - 0.5f0
    ix = floor(fx); iy = floor(fy)
    tx = fx - ix;   ty = fy - iy
    # Repeat wrap on the integer indices too (-1 → s, s → 1).
    row0 = mod(Int(ix),   s[1]) + 1
    row1 = mod(Int(ix)+1, s[1]) + 1
    col0 = mod(Int(iy),   s[2]) + 1
    col1 = mod(Int(iy)+1, s[2]) + 1
    @inbounds v00 = data[row0, col0]
    @inbounds v10 = data[row1, col0]
    @inbounds v01 = data[row0, col1]
    @inbounds v11 = data[row1, col1]
    w00 = (1f0 - tx) * (1f0 - ty)
    w10 = tx * (1f0 - ty)
    w01 = (1f0 - tx) * ty
    w11 = tx * ty
    return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11
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

