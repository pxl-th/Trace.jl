module Trace

using Parameters: @with_kw
using LinearAlgebra
using StaticArrays
using GeometryBasics

GeometryBasics.@fixed_vector Normal StaticVector
const Normal3f0 = Normal{3, Float32}

Maybe{T} = Union{T, Nothing}
maybe_copy(v::Maybe)::Maybe = v isa Nothing ? v : copy(v)

@inline sum_mul(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

function find_interval(size::Int64, predicate::Function)
    first, len = 0, size
    while len > 1
        half = len >> 1
        middle = first + half
        if predicate(middle)
            first = middle + 1
            len -= half + 1
        else
            len = half
        end
    end
    clamp(first, 1, size - 1)
end

function partition!(x::Vector, range::UnitRange, predicate::Function)
    left = range[1]
    for i in range
        if left != i && predicate(x[i])
            x[i], x[left] = x[left], x[i]
            left += 1
        end
    end
    left
end

function coordinate_system(v1::Vec3f0, v2::Vec3f0)
    if abs(v1[1]) > abs(v1[2])
        v2 = typeof(v2)(-v1[3], 0, v1[1]) / sqrt(v1[1] * v1[1] + v1[3] * v1[3])
    else
        v2 = typeof(v2)(0, v1[3], -v1[2]) / sqrt(v1[2] * v1[2] + v1[3] * v1[3])
    end
    v1, v2, v1 × v2
end

function spherical_direction(sin_θ::Float32, cos_θ::Float32, ϕ::Float32)
    Vec3f0(sin_θ * cos(ϕ), sin_θ * sin(ϕ), cos_θ)
end
function spherical_direction(
    sin_θ::Float32, cos_θ::Float32, ϕ::Float32,
    x::Vec3f0, y::Vec3f0, z::Vec3f0,
)
    sin_θ * cos(ϕ) * x + sin_θ * sin(ϕ) * y + cos_θ * z
end

spherical_θ(v::Vec3f0) = clamp(v[3], -1, 1) |> acos
function spherical_ϕ(v::Vec3f0)
    p = atan(v[2], v[1])
    p < 0 ? p + 2 * π : p
end


"""
Flip normal `n` so that it lies in the same hemisphere as `v`.
"""
face_forward(n, v) = (n ⋅ v) < 0 ? -n : n

include("ray.jl")
include("bounds.jl")
include("transformations.jl")

# TODO AnimatedTransform, AnimatedBounds
# TODO Medium & add it to structs

abstract type AbstractShape end
abstract type Primitive end

mutable struct Interaction
    p::Point3f0
    time::Float32
    wo::Vec3f0  # Negative direction of ray (for ray-shape interactions).
    n::Normal3f0  # Surface normal at the point.
end

mutable struct ShadingInteraction
    n::Normal3f0
    ∂p∂u::Vec3f0
    ∂p∂v::Vec3f0
    ∂n∂u::Normal3f0
    ∂n∂v::Normal3f0
end

mutable struct SurfaceInteraction{S <: AbstractShape}
    core::Interaction
    shading::ShadingInteraction
    uv::Point2f0

    ∂p∂u::Vec3f0
    ∂p∂v::Vec3f0
    ∂n∂u::Normal3f0
    ∂n∂v::Normal3f0

    shape::Maybe{S}
    primitive::Maybe{P} where P <: Primitive

    ∂u∂x::Float32
    ∂u∂y::Float32
    ∂v∂x::Float32
    ∂v∂y::Float32
    ∂p∂x::Vec3f0
    ∂p∂y::Vec3f0

    function SurfaceInteraction(
        p::Point3f0, time::Float32, wo::Vec3f0, uv::Point2f0,
        ∂p∂u::Vec3f0, ∂p∂v::Vec3f0, ∂n∂u::Normal3f0, ∂n∂v::Normal3f0,
        shape::Maybe{S} = nothing, primitive::Maybe{P} = nothing,
    ) where S <: AbstractShape where P <: Primitive
        n = ∂p∂u × ∂p∂v
        if !(shape isa Nothing) && (shape.core.reverse_orientation ⊻ shape.core.transform_swaps_handedness)
            n *= -1
        end

        core = Interaction(p, time, wo, n)
        shading = ShadingInteraction(n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
        new{typeof(shape)}(
            core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v, shape, primitive,
            0f0, 0f0, 0f0, 0f0, Vec3f0(0f0), Vec3f0(0f0),
        )
    end
end

function set_shading_geometry!(
    i::SurfaceInteraction, tangent::Vec3f0, bitangent::Vec3f0,
    ∂n∂u::Normal3f0, ∂n∂v::Normal3f0, orientation_is_authoritative::Bool,
)
    i.shading.n = normalize(tangent × bitangent)
    if !(i.shape isa Nothing) && (i.shape.core.reverse_orientation ⊻ i.shape.core.transform_swaps_handedness)
        i.shading.n *= -1
    end
    if orientation_is_authoritative
        i.core.n = face_forward(i.core.n, i.shading.n)
    else
        i.shading.n = face_forward(i.shading.n, i.core.n)
    end

    i.shading.∂p∂u = tangent
    i.shading.∂p∂v = bitangent
    i.shading.∂n∂u = ∂n∂u
    i.shading.∂n∂v = ∂n∂v
end

is_surface_interaction(i::Interaction) = i.n != Normal3f0(0)

"""
Compute partial derivatives needed for computing sampling rates
for things like texture antialiasing.
"""
function compute_differentials!(si::SurfaceInteraction, ray::RayDifferentials)
    if !ray.has_differentials
        si.∂u∂x = si.∂v∂x = 0f0
        si.∂u∂y = si.∂v∂y = 0f0
        si.∂p∂x = si.∂p∂y = Vec3f0(0f0)
        return
    end
    # Estimate screen change in p and (u, v).
    # Compute auxiliary intersection points with plane.
    d = -(si.core.n ⋅ si.core.p)
    tx = (-(si.core.n ⋅ ray.rx_origin) - d) / (si.core.n ⋅ ray.rx_direction)
    ty = (-(si.core.n ⋅ ray.ry_origin) - d) / (si.core.n ⋅ ray.ry_direction)
    px = ray.rx_origin + tx * ray.rx_direction
    py = ray.ry_origin + ty * ray.ry_direction

    si.∂p∂x = px - si.core.p
    si.∂p∂y = py - si.core.p
    # Compute (u, v) offsets at auxiliary points.
    # Choose two dimensions for ray offset computation.
    n = abs.(si.core.n)
    if n[1] > n[2] && n[1] > n[3]
        dim = Point2f0(2, 3)
    elseif n[2] > n[3]
        dim = Point2f0(1, 3)
    else
        dim = Point2f0(1, 2)
    end
    # Initialization for offset computation.
    a = Mat2f0(dim[1], dim[1], dim[2], dim[2])
    bx = Point2f0(px[dim[1]] - si.core.p[dim[1]], px[dim[2]] - si.core.p[dim[2]])
    by = Point2f0(py[dim[1]] - si.core.p[dim[1]], py[dim[2]] - si.core.p[dim[2]])
    sx = a \ bx
    sy = a \ by

    si.∂u∂x, si.∂v∂x = any(isnan.(sx)) ? (0f0, 0f0) : sx
    si.∂u∂y, si.∂v∂y = any(isnan.(sy)) ? (0f0, 0f0) : sy
end

include("shapes/Shape.jl")
include("primitive.jl")
include("accel/bvh.jl")
include("spectrum.jl")

include("filter.jl")
include("film.jl")
include("reflection.jl")

include("camera/camera.jl")
# include("sampler/sampler.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("materials/bsdf.jl")

"""
TODO
- assert that t_max in intersect methods >= 0
- test if bvh contains duplicates
"""

# @info from_RGB(SampledSpectrum, Point3f0(1f0, 0f0, 0f0), Illuminant)
# @info from_RGB(SampledSpectrum, Point3f0(1f0, 0f0, 0f0), Reflectance)
# @info from_XYZ(SampledSpectrum, Point3f0(0.5f0, 0f0, 0.5f0))
# @info from_RGB(RGBSpectrum, Point3f0(0.5f0, 0f0, 0.5f0))
# @info from_XYZ(RGBSpectrum, Point3f0(0.5f0, 0f0, 0.5f0))

# @info Point3f0(1.0, 0.0, 0.0) |> RGB_to_XYZ
# @info Point3f0(1.0, 0.0, 0.0) |> RGB_to_XYZ |> XYZ_to_RGB

end
