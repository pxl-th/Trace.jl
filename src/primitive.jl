struct GeometricPrimitive{T <: AbstractShape} <: Primitive
    shape::T
end

function intersect!(p::GeometricPrimitive, ray::Ray)
    intersects, t_hit, interaction = intersect(p, ray)
    !intersects && return false, nothing
    ray.t_max = t_hit
    interaction.primitive = p
    true, interaction
end

intersect_p(p::GeometricPrimitive, ray::Ray) = intersect_p(p.shape, ray)
world_bound(p::GeometricPrimitive) = p.shape |> world_bound
