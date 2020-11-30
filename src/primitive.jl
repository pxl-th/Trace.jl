struct GeometricPrimitive{T <: AbstractShape} <: Primitive
    shape::T
    material::Maybe{M} where M <: Material

    function GeometricPrimitive(
        shape::T, material::Maybe{M} = nothing,
    ) where {T <: AbstractShape, M <: Material}
        new{T}(shape, material)
    end
end

function intersect!(p::GeometricPrimitive{T}, ray::Ray) where T <: AbstractShape
    intersects, t_hit, interaction = intersect(p.shape, ray)
    !intersects && return false, nothing
    ray.t_max = t_hit
    interaction.primitive = p
    true, interaction
end

intersect_p(p::GeometricPrimitive, ray::Ray) = intersect_p(p.shape, ray)
world_bound(p::GeometricPrimitive) = p.shape |> world_bound
