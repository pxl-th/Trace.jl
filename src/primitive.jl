struct GeometricPrimitive{T<:AbstractShape,M<:UberMaterial} <: Primitive
    shape::T
    material::M

    function GeometricPrimitive(
        shape::T, material::UberMaterial=NoMaterial(),
    ) where {T<:AbstractShape}
        new{T, typeof(material)}(shape, material)
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
        pool, p::GeometricPrimitive, si::SurfaceInteraction,
        allow_multiple_lobes::Bool, transport::UInt8,
    )
    bsdf = nothing
    if p.material.type !== NO_MATERIAL
        bsdf = p.material(pool, si, allow_multiple_lobes, transport)
    end
    @real_assert (si.core.n ⋅ si.shading.n) ≥ 0f0
    return bsdf
end
