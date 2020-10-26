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

struct Interaction
    p::Point3f0
    time::Float32
    p_error::Vec3f0
    wo::Vec3f0  # Negative direction of ray (for ray-shape interactions).
    n::Normal3f0  # Surface normal at the point.
end

is_surface_interaction(i::Interaction) = i.n != Normal3f0(0)

include("shape.jl")

end
