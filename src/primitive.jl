struct GeometricPrimitive{T<:AbstractShape} <: Primitive
    shape::T
    material::Maybe{M} where M<:Material

    function GeometricPrimitive(
        shape::T, material::Maybe{M} = nothing,
    ) where {T<:AbstractShape,M<:Material}
        new{T}(shape, material)
    end
end

function intersect!(
    pool, p::GeometricPrimitive{T}, ray::Union{Ray,RayDifferentials},
) where T<:AbstractShape
    shape = p.shape
    intersects, t_hit, interaction = intersect(pool, shape, ray)
    !intersects && return false, interaction
    ray.t_max = t_hit
    return true, interaction
end

@inline function intersect_p(
    pool, p::GeometricPrimitive, ray::Union{Ray,RayDifferentials},
)
    intersect_p(pool, p.shape, ray)
end
@inline world_bound(p::GeometricPrimitive) = world_bound(p.shape)

function compute_scattering!(
    p::GeometricPrimitive, si::SurfaceInteraction,
    allow_multiple_lobes::Bool, ::Type{T},
) where T<:TransportMode
    if !(p.material isa Nothing)
        bsdf = p.material(si, allow_multiple_lobes, T)
    end
    @real_assert (si.core.n ⋅ si.shading.n) ≥ 0f0
    return bsdf
end
