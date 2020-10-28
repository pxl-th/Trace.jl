module Trace

using Parameters: @with_kw
using LinearAlgebra
using StaticArrays
using GeometryBasics

GeometryBasics.@fixed_vector Normal StaticVector
const Normal3f0 = Normal{3, Float32}

Maybe{T} = Union{T, Nothing}
maybe_copy(v::Maybe)::Maybe = v isa Nothing ? v : copy(v)

function coordinate_system(v1::Vec3f0, v2::Vec3f0)
    if abs(v1[1]) > abs(v1[2])
        v2 = typeof(v2)(-v1[3], 0, v1[1]) / sqrt(v1[1] * v1[1] + v1[3] * v1[3])
    else
        v2 = typeof(v2)(0, v1[3], -v1[2]) / sqrt(v1[2] * v1[2] + v1[3] * v1[3])
    end
    v1, v2, v1 × v2
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

struct SurfaceInteraction{S}
    core::Interaction
    shading::ShadingInteraction
    uv::Point2f0

    ∂p∂u::Vec3f0
    ∂p∂v::Vec3f0
    ∂n∂u::Normal3f0
    ∂n∂v::Normal3f0

    shape::Union{Nothing, S}

    function SurfaceInteraction(
        p::Point3f0, time::Float32, wo::Vec3f0, uv::Point2f0,
        ∂p∂u::Vec3f0, ∂p∂v::Vec3f0, ∂n∂u::Normal3f0, ∂n∂v::Normal3f0,
        shape::Union{Nothing, S} = nothing,
    ) where S <: AbstractShape
        n = ∂p∂u × ∂p∂v
        if !(shape isa Nothing) && (shape.core.reverse_orientation ⊻ shape.core.transform_swaps_handedness)
            n *= -1
        end

        core = Interaction(p, time, wo, n)
        shading = ShadingInteraction(n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
        new{typeof(shape)}(core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v, shape)
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

include("shapes/Shape.jl")

# tm = create_triangle_mesh(
#     ShapeCore(translate(Vec3f0(0)), translate(Vec3f0(0)), false),
#     1, UInt32[1, 2, 3],
#     3, [Point3f0(-1, -1, 2), Point3f0(0, 1, 2), Point3f0(1, -1, 2)],
#     [Normal3f0(0, 0, -1), Normal3f0(0, 0, -1), Normal3f0(0, 0, -1)],
# )
# t = tm[1]

# r = Ray(o=Point3f0(0), d=Vec3f0(0, 0, 1))

# @info object_bound(t)
# @info world_bound(t)
# i, t_hit, interaction = intersect(t, r)
# @info i, r(t_hit)
# @info interaction

# rot = rotate_x(45f0)
# v = Vec3f0(1)
# n = Normal3f0(1)
# Normal3f0(v)
# convert(Normal3f0, Point3f0(0))
# @info n ⋅ n
# @info n × n

# rot(v)
# rot(n)

end
