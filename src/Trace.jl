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
function partition!(x, range, predicate)
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

mutable struct SurfaceInteraction{S <: AbstractShape, P <: Maybe{Primitive}}
    core::Interaction
    shading::ShadingInteraction
    uv::Point2f0

    ∂p∂u::Vec3f0
    ∂p∂v::Vec3f0
    ∂n∂u::Normal3f0
    ∂n∂v::Normal3f0

    shape::Maybe{S}
    primitive::Maybe{P}

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
        new{typeof(shape), typeof(primitive)}(
            core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v, shape, primitive,
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

include("shapes/Shape.jl")
include("primitive.jl")
include("accel/bvh.jl")

# tm = create_triangle_mesh(
#     ShapeCore(translate(Vec3f0(0)), translate(Vec3f0(0)), false),
#     1, UInt32[1, 2, 3],
#     3, [Point3f0(-1, -1, 2), Point3f0(0, 1, 2), Point3f0(1, -1, 2)],
#     [Normal3f0(0, 0, -1), Normal3f0(0, 0, -1), Normal3f0(0, 0, -1)],
# )
# t = tm[1]

# r = Ray(o=Point3f0(0), d=Vec3f0(0, 0, 1))

# @info area(t)
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

core = ShapeCore(translate(Vec3f0(10, 0, 0)), translate(Vec3f0(-10, 0, 0)), false)
core2 = ShapeCore(translate(Vec3f0(0, 10, 0)), translate(Vec3f0(0, -10, 0)), false)
core3 = ShapeCore(translate(Vec3f0(0, 0, 10)), translate(Vec3f0(0, 0, -10)), false)
s = Sphere(core, 1f0, -1f0, 1f0, 360f0)
s2 = Sphere(core2, 1f0, -1f0, 1f0, 360f0)
s3 = Sphere(core3, 1f0, -1f0, 1f0, 360f0)
p1 = GeometricPrimitive(s)
p2 = GeometricPrimitive(s2)
p3 = GeometricPrimitive(s3)
bvh = BVHAccel{HLBVH}([p1, p2, p3], 1)
@info bvh.root.bounds
@info bvh.root.children[1].bounds
@info bvh.root.children[1].children[1].bounds
@info bvh.root.children[1].children[2].bounds
@info bvh.root.children[2].bounds
# TODO implement Primitive interface for this (world_bound, etc.)
# bvh2 = BVHAccel{SAH}([p1, bvh], 1)

end
