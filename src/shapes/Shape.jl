struct ShapeCore
    object_to_world::Transformation
    world_to_object::Transformation
    reverse_orientation::Bool
    transform_swaps_handedness::Bool

    function ShapeCore(
        object_to_world::Transformation, world_to_object::Transformation,
        reverse_orientation::Bool,
    )
        new(
            object_to_world, world_to_object, reverse_orientation,
            object_to_world |> swaps_handedness,
        )
    end
end

# Interface.
function area(::AbstractShape)::Float32 end
function object_bound(::AbstractShape)::Bounds3 end
function world_bound(s::AbstractShape)::Bounds3
    s |> object_bound |> s.core.object_to_world
end
# function intersect(
#     ::AbstractShape, ::Ray, test_alpha_texture::Bool = true,
# )::Tuple{Float32, SurfaceInteraction} end

include("sphere.jl")
include("triangle_mesh.jl")
