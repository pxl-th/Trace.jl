module Trace

using Parameters: @with_kw
using LinearAlgebra
using GeometryBasics

function coordinate_system(v1::Vec3f0, v2::Vec3f0)
    if abs(v1[1]) > abs(v1[2])
        v2 = typeof(v2)(-v1[3], 0, v1[1]) / sqrt(v1[1] * v1[1] + v1[3] * v1[3])
    else
        v2 = typeof(v2)(0, v1[3], -v1[2]) / sqrt(v1[2] * v1[2] + v1[3] * v1[3])
    end
    v1, v2, v1 × v2
end

Normal3f0 = Vec3f0

"""
Flip normal `n` so that it lies in the same hemisphere as `v`.
"""
function face_forward(n::Normal3f0, v::Vec3f0)
    (n ⋅ v) < 0 ? -n : n
end

include("ray.jl")
include("bounds.jl")
include("transformations.jl")

# TODO AnimatedTransform, AnimatedBounds
# TODO SurfaceInteraction
# TODO Medium & add it to structs

abstract type AbstractShape end

struct Interaction
    p::Point3f0
    time::Float32
    wo::Vec3f0  # Negative direction of ray (for ray-shape interactions).
    n::Normal3f0  # Surface normal at the point.
end

struct ShadingInteraction
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

is_surface_interaction(i::Interaction) = i.n != Normal3f0(0)

include("shape.jl")

# core = ShapeCore(translate(Vec3f0(0, 2, 0)), translate(Vec3f0(0, -2, 0)), false)
# s = Sphere(core, 1f0, -1f0, 1f0, 360f0)
# r = Ray(o=Point3f0(0, 0, 0), d=Vec3f0(0, 1, 0))

# intersects, t, interaction = intersect(s, r, false)
# @info "Intersects: $intersects || $(intersect_p(s, r, false)) @ $t"
# @info "World interaction $(r(t))"
# @info "Interaction @ $(interaction.core.p)"
# @info "Area: $(area(s))"

end
