struct ShapeCore
    object_to_world::Transformation
    world_to_object::Transformation
    reverse_orientation::Bool
    transform_swaps_handedness::Bool

    function ShapeCore(
        object_to_world::Transformation, reverse_orientation::Bool,
    )
        new(
            object_to_world, object_to_world |> inv, reverse_orientation,
            object_to_world |> swaps_handedness,
        )
    end
end

function world_bound(s::AbstractShape)::Bounds3
    s |> object_bound |> s.core.object_to_world
end

include("sphere.jl")
include("triangle_mesh.jl")
