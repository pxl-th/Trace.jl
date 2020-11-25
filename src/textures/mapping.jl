abstract type Mapping2D end
abstract type Mapping3D end

"""
Simplest mapping uses `(u, v)` coordinates from the `SurfaceInteraction`
to compute texture coordinates and can be offset and scaled with user-supplied
values in each dimension.
"""
struct UVMapping2D <: Mapping2D
    su::Float32
    sv::Float32
    du::Float32
    dv::Float32
end

"""
Given surface interaction `si` at the shading point,
return `(s, t)` texture coordinates and estimated changes in `(s, t)` w.r.t.
pixel `x` & `y` coordinates.

- `m::UVMapping2D`: UVMapping with offset & scale parameters.
- `si::SurfaceInteraction`: SurfaceInteraction at the shading point.

# Returns
    `Tuple{Point2f0, Vec2f0, Vec2f0}`:
        Texture coordinates at the shading point, estimated changes
        in `(s, t)` w.r.t. pixel `x` & `y` coordinates.
"""
function map(m::UVMapping2D, si::SurfaceInteraction)::Tuple{Point2f0, Vec2f0, Vec2f0}
    ∂st∂x = Vec2f0(m.su * si.∂u∂x, m.sv * si.∂v∂x)
    ∂st∂y = Vec2f0(m.su * si.∂u∂y, m.sv * si.∂v∂y)
    t = Point2f0(m.su * si.uv[1] + m.du, m.sv * si.uv[2] + m.dv)
    t, ∂st∂x, ∂st∂y
end


struct TransformMapping3D <: Mapping3D
    world_to_texture::Transformation
end

function map(m::TransformMapping3D, si::SurfaceInteraction)::Tuple{Point3f0, Vec3f0, Vec3f0}
    ∂p∂x = si.∂p∂x |> m.world_to_texture
    ∂p∂y = si.∂p∂y |> m.world_to_texture
    t = si.core.p |> m.world_to_texture
    t, ∂p∂x, ∂p∂y
end
