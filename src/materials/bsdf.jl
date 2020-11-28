"""
Maximum allowed number of BxDF components in BSDF.
"""
const MAX_BxDF = UInt8(8)

mutable struct BSDF
    """
    Relative index of refraction over the boundary.
    For opaque surfaces it is not used and should be 1.
    """
    η::Float32
    """
    Geometric normal defined by the surface geometry.
    """
    ng::Normal3f0
    """
    Shading normal defined by per-vertex normals and/or bump-mapping.
    These normals will generally define different hemispheres for
    integrating incident illumination to computer surface reflection.
    """
    ns::Normal3f0
    """
    Component of the orhonormal coordinates system with the
    shading normal as one of the axes.
    """
    ss::Vec3f0
    """
    Component of the orhonormal coordinates system with the
    shading normal as one of the axes.
    """
    ts::Vec3f0
    """
    Current number of BxDFs (≤ 8).
    """
    n_bxdfs::UInt8
    """
    Individual BxDF components. Maximum allowed number of components is 8.
    """
    bxdfs::Vector{B} where B <: BxDF

    function BSDF(si::SurfaceInteraction, η::Float32 = 1f0)
        ns = si.shading.n
        ng = si.core.n
        ss = normalize(si.shading.∂p∂u)
        ts = ns × ss
        new(
            η, ng, ns, ss, ts, UInt8(0),
            Vector{B where B <: BxDF}(undef, MAX_BxDF),
        )
    end
end

function add!(b::BSDF, x::B) where B <: BxDF
    @assert b.n_bxdfs < MAX_BxDF
    b.n_bxdfs += 1
    b.bxdfs[b.n_bxdfs] = x
end

"""
Given the orthonormal vectors s, t, n in world space, the matrix `M`
that transforms vectors in world space to local reflection space is:
    sx, sy, sz
    tx, ty, tz
    nx, ny, nz

Since it is an orthonormal matrix, its inverse is its transpose.
"""
world_to_local(b::BSDF, v::Vec3f0) = Vec3f0(v ⋅ b.ss, v ⋅ b.ts, v ⋅ b.ns)
local_to_world(b::BSDF, v::Vec3f0) = Mat3f0(b.ss..., b.ts..., b.ns) * n

function (b::BSDF)(wo_world::Vec3f0, wi_world::Vec3f0, f::BxDFTypes)
    wi = world_to_local(b, wi_world)
    wo = world_to_local(b, wo_world)
    reflect = ((wi_world ⋅ b.ng) * (wo_world ⋅ b.ng)) > 0

    output = RGBSpectrum(0f0) # TODO assumes that default is RGBSpectrum
    for i in 1:b.n_bxdfs
        bxdf = b.bxdfs[i]
        if ((bxdf & flags) && (
            (reflect && (bxdf & BSDF_REFLECTION))
            || (!reflect && (bxdf & BSDF_TRANSMISSION))
        ))
            output += bxdf(wo, wi)
        end
    end
    output
end
