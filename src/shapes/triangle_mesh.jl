struct TriangleMesh
    vertices::Vector{Point3f}
    # For the i-th triangle, its 3 vertex positions are:
    # [vertices[indices[3 * i + j]] for j in 0:2].
    indices::Vector{UInt32}
    # Optional normal vectors, one per vertex.
    normals::Vector{Normal3f}
    # Optional tangent vectors, one per vertex.
    tangents::Vector{Vec3f}
    # Optional parametric (u, v) values, one for each vertex.
    uv::Vector{Point2f}

    function TriangleMesh(
            object_to_world::Transformation,
            indices::Vector{UInt32},
            vertices::Vector{Point3f},
            normals::Vector{Normal3f} = Normal3f[],
            tangents::Vector{Vec3f} = Vec3f[],
            uv::Vector{Point2f} = Point2f[],
        )
        vertices = object_to_world.(vertices)
        return new(
            vertices,
            copy(indices), copy(normals),
            copy(tangents), copy(uv),
        )
    end
end

struct Triangle <: AbstractShape
    core::ShapeCore
    mesh::TriangleMesh
    # Id to the first index of triangle in mesh.indices.
    # ids = [mesh.indices[i + j] for j in 0:2]
    i::UInt32

    # i maps from 0 based indexing to 1 based.
    function Triangle(core::ShapeCore, mesh::TriangleMesh, i::Integer)
        new(core, mesh, i * 3 + 1)
    end
end

@inline function shading_normal(shape::Triangle, core_n, shading_n)
    if !isempty(shape.mesh.normals)
        core_n = face_forward(
            core_n, shading_n,
        )
    elseif shape.core.reverse_orientation ⊻ shape.core.transform_swaps_handedness
        core_n = shading_n = -core_n
    end
    return core_n, shading_n
end

function create_triangle_mesh(
    core::ShapeCore,
    indices::Vector{UInt32},
    vertices::Vector{Point3f},
    normals::Vector{Normal3f} = Normal3f[],
    tangents::Vector{Vec3f} = Vec3f[],
    uv::Vector{Point2f} = Point2f[],
)
    mesh = TriangleMesh(
        core.object_to_world, indices, vertices,
        normals, tangents, uv,
    )
    ntriangles = length(indices) ÷ 3
    [Triangle(core, mesh, i-1) for i in 1:ntriangles]
end

function create_triangle_mesh(mesh::GeometryBasics.Mesh, core::ShapeCore)
    fs = decompose(TriangleFace{UInt32}, mesh)
    vertices = decompose(Point3f, mesh)
    normals = Normal3f.(decompose_normals(mesh))
    uvs = Point2f.(GeometryBasics.decompose_uv(mesh))
    indices = collect(reinterpret(UInt32, fs))
    mesh = TriangleMesh(
        core.object_to_world, indices, vertices,
        normals, Vec3f[], uvs,
    )
    ntriangles = length(fs)
    [Triangle(core, mesh, i) for i in UnitRange{UInt32}(0:ntriangles-1)]
end


function area(t::Triangle)
    vs = vertices(t)
    0.5f0 * norm((vs[2] - vs[1]) × (vs[3] - vs[1]))
end

function is_degenerate(vs::AbstractVector{Point3f})::Bool
    v = (vs[3] - vs[1]) × (vs[2] - vs[1])
    (v ⋅ v) ≈ 0f0
end

function vertices(t::Triangle)
    @inbounds @SVector Point3f[t.mesh.vertices[t.mesh.indices[t.i+j]] for j in 0:2]
end
function normals(t::Triangle)
    @inbounds @SVector Normal3f[t.mesh.normals[t.mesh.indices[t.i+j]] for j in 0:2]
end
function tangents(t::Triangle)
    @inbounds @SVector Vec3f[t.mesh.tangents[t.mesh.indices[t.i+j]] for j in 0:2]
end
function uvs(t::Triangle)
    if isempty(t.mesh.uv) &&
        return @SVector [Point2f(0), Point2f(1, 0), Point2f(1, 1)]
    end
    idxs = t.mesh.indices
    @inbounds @SVector [t.mesh.uv[idxs[t.i+j]] for j in 0:2]
end

function _edge_function(vs)
    Point3f(
        vs[2][1] * vs[3][2] - vs[2][2] * vs[3][1],
        vs[3][1] * vs[1][2] - vs[3][2] * vs[1][1],
        vs[1][1] * vs[2][2] - vs[1][2] * vs[2][1],
    )
end

object_bound(t::Triangle) = mapreduce(
    v -> Bounds3(t.core.world_to_object(v)),
    ∪, vertices(t),
)
world_bound(t::Triangle) = reduce(∪, Bounds3.(vertices(t)))

function _to_ray_coordinate_space(
        vertices::AbstractVector{Point3f}, ray::Union{Ray,RayDifferentials},
    )
    # Compute permutation.
    kz = argmax(abs.(ray.d))
    kx = kz + 1
    kx == 4 && (kx = 1)
    ky = kx + 1
    ky == 4 && (ky = 1)
    permutation = @SVector [kx, ky, kz]
    # Permute ray direction.
    d = ray.d[permutation]
    # Compute shear.
    denom = 1f0 / d[3]
    shear = Point3f(-d[1] * denom, -d[2] * denom, denom)
    # Translate, apply permutation and shear to vertices.
    rkz = ray.o[kz]
    tvs = ntuple(3) do i
        v = vertices[i]
        vo = (v-ray.o)[permutation]
        return vo + Point3f(
            shear[1] * (v[kz] - rkz),
            shear[2] * (v[kz] - rkz),
            0.0f0,
        )
    end
    return SVector(tvs), shear
end

function ∂p(
        ::Triangle, vs::AbstractVector{Point3f}, uv::AbstractVector{Point2f},
    )::Tuple{Vec3f,Vec3f,Vec3f,Vec3f}

    # Compute deltas for partial derivative matrix.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δp_13, δp_23 = Vec3f(vs[1] - vs[3]), Vec3f(vs[2] - vs[3])
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    if det ≈ 0
        v = normalize((vs[3] - vs[1]) × (vs[2] - vs[1]))
        _, ∂p∂u, ∂p∂v = coordinate_system(Vec3f(v))
        return ∂p∂u, ∂p∂v, δp_13, δp_23
    end
    inv_det = 1f0 / det
    ∂p∂u = Vec3f(δuv_23[2] * δp_13 - δuv_13[2] * δp_23) * inv_det
    ∂p∂v = Vec3f(-δuv_23[1] * δp_13 + δuv_13[1] * δp_23) * inv_det
    ∂p∂u, ∂p∂v, δp_13, δp_23
end

function ∂n(
        t::Triangle, uv::AbstractVector{Point2f},
    )::Tuple{Normal3f,Normal3f}

    t.mesh.normals isa Nothing && return Normal3f(0), Normal3f(0)
    t_normals = normals(t)
    # Compute deltas for partial detivatives of normal.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δn_13, δn_23 = t_normals[1] - t_normals[3], t_normals[2] - t_normals[3]
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    det ≈ 0 && return Normal3f(0), Normal3f(0)

    inv_det = 1f0 / det
    ∂n∂u = (δuv_23[2] * δn_13 - δuv_13[2] * δn_23) * inv_det
    ∂n∂v = (-δuv_23[1] * δn_13 + δuv_13[1] * δn_23) * inv_det
    ∂n∂u, ∂n∂v
end

function _init_triangle_shading_geometry(
        t::Triangle, si::SurfaceInteraction,
        barycentric::Point3f, uv::AbstractVector{Point2f},
    )
    !(!isempty(t.mesh.normals) || !isempty(t.mesh.tangents)) && return si
    # Initialize triangle shading geometry.
    # Compute shading normal, tangent & bitangent.
    ns = si.core.n
    if !isempty(t.mesh.normals)
        ns = normalize(sum_mul(barycentric, normals(t)))
    end
    if !isempty(t.mesh.tangents)
        ss = normalize(sum_mul(barycentric, tangents(t)))
    else
        ss = normalize(si.∂p∂u)
    end
    ts = ns × ss
    if (ts ⋅ ts) > 0
        ts = Vec3f(normalize(ts))
        ss = Vec3f(ts × ns)
    else
        _, ss, ts = coordinate_system(ns)
    end
    ∂n∂u, ∂n∂v = ∂n(t, uv)

    return set_shading_geometry(t, si, ss, ts, ∂n∂u, ∂n∂v, true)
end

function create_surface_interaction(
        t, normal, ray, hitpoint, uvs, uv, barycentric, ∂p∂u, ∂p∂v,
        orientation_is_authoritative, reverse_normal)

    ∂n∂u = Normal3f(0)
    ∂n∂v = Normal3f(0)
    time = ray.time
    wo = -ray.d
    core_n = normal
    shading_n = normal
    shading_∂p∂u = ∂p∂u
    shading_∂p∂v = ∂p∂v
    shading_∂n∂u = ∂n∂u
    shading_∂n∂v = ∂n∂v

    if t.mesh isa Nothing || !(!isempty(t.mesh.normals) || !isempty(t.mesh.tangents))
        return SurfaceInteraction(
            Interaction(hitpoint, time, wo, core_n),
            ShadingInteraction(shading_n, shading_∂p∂u, shading_∂p∂v, shading_∂n∂u, shading_∂n∂v),
            uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
            0.0f0, 0.0f0, 0.0f0, 0.0f0, Vec3f(0.0f0), Vec3f(0.0f0)
        )
    end

    # Initialize triangle shading geometry.
    # Compute shading normal, tangent & bitangent.

    ns = core_n
    if !isempty(t.mesh.normals)
        ns = normalize(sum_mul(barycentric, normals(t)))
    end
    if !isempty(t.mesh.tangents)
        ss = normalize(sum_mul(barycentric, tangents(t)))
    else
        ss = normalize(∂p∂u)
    end
    ts = ns × ss
    if (ts ⋅ ts) > 0
        ts = Vec3f(normalize(ts))
        ss = Vec3f(ts × ns)
    else
        _, ss, ts = coordinate_system(ns)
    end
    ∂n∂u, ∂n∂v = ∂n(t, uvs)

    shading_n = normalize(∂n∂v × ∂n∂v)
    if reverse_normal
        shading_n *= -1
    end
    if orientation_is_authoritative
        core_n = face_forward(core_n, shading_n)
    else
        shading_n = face_forward(shading_n, core_n)
    end

    shading_∂p∂u = ∂n∂u
    shading_∂p∂v = ∂n∂v
    shading_∂n∂u = ∂n∂u
    shading_∂n∂v = ∂n∂v

    # Ensure correct orientation of the geometric normal.
    if !isempty(t.mesh.normals)
        core_n = face_forward(core_n, shading_n)
    elseif t.core.reverse_orientation ⊻ t.core.transform_swaps_handedness
        core_n = shading_n = -core_n
    end
    return SurfaceInteraction(

        Interaction(hitpoint, time, wo, core_n),

        ShadingInteraction(shading_n, shading_∂p∂u, shading_∂p∂v, shading_∂n∂u, shading_∂n∂v),
        uv,

        ∂p∂u,
        ∂p∂v,
        ∂n∂u,
        ∂n∂v,

        0f0, 0f0, 0f0, 0f0, Vec3f(0f0), Vec3f(0f0)
    )
end


@inline function intersect(
        t::Triangle, ray::Union{Ray,RayDifferentials}, ::Bool = false,
    )::Tuple{Bool,Float32,SurfaceInteraction}

    vs = vertices(t)
    si = SurfaceInteraction()
    is_degenerate(vs) && return false, 0.0f0, si
    t_vs, shear = _to_ray_coordinate_space(vs, ray)

    # Compute edge function coefficients.

    edges = _edge_function(t_vs)
    if iszero(edges) # Fall-back to double precision.
        edges = _edge_function(Float64.(t_vs))
    end
    # Perform triangle edge & determinant tests.
    # Point is inside a triangle if all edges have the same sign.
    any(edges .< 0) && any(edges .> 0) && return false, 0f0, si
    det = sum(edges)
    det ≈ 0 && return false, 0f0, si
    # Compute scaled hit distance to triangle.
    shear_z = shear[3]
    t_scaled = (
        edges[1] * t_vs[1][3] * shear_z
        + edges[2] * t_vs[2][3] * shear_z
        + edges[3] * t_vs[3][3] * shear_z
    )
    # Test against t_max range.
    det < 0 && (t_scaled >= 0 || t_scaled < ray.t_max * det) &&
        return false, 0f0, si
    det > 0 && (t_scaled <= 0 || t_scaled > ray.t_max * det) &&
        return false, 0f0, si
    # Compute barycentric coordinates and t value for triangle intersection.
    inv_det = 1f0 / det
    barycentric = edges .* inv_det
    t_hit = t_scaled * inv_det
    # TODO check that t_hit > 0
    uv = uvs(t)
    ∂p∂u, ∂p∂v, δp_13, δp_23 = ∂p(t, vs, uv)
    # Interpolate (u, v) paramteric coordinates and hit point.
    hit_point = sum_mul(barycentric, vs)
    uv_hit = sum_mul(barycentric, uv)
    normal = normalize(δp_13 × δp_23)

    si = SurfaceInteraction(
        normal, hit_point, ray.time, -ray.d, uv_hit,
        ∂p∂u, ∂p∂v, Normal3f(0), Normal3f(0)
    )
    if t.mesh isa Nothing
        return true, t_hit, si
    end

    si = _init_triangle_shading_geometry(t, si, barycentric, uv)
    # TODO test against alpha texture if present.
    return true, t_hit, si
end

@inline function intersect_p(
        t::Triangle, ray::Union{Ray,RayDifferentials}, ::Bool = false,
    )::Bool

    vs = vertices(t)
    is_degenerate(vs) && return false
    t_vs, shear = _to_ray_coordinate_space(vs, ray)
    # Compute edge function coefficients.
    edges = _edge_function(t_vs)
    if iszero(edges) # Fall-back to double precision.
        edges = _edge_function((x -> x .|> Float64).(t_vs))
    end
    # Perform triangle edge & determinant tests.
    # Point is inside a triangle if all edges have the same sign.
    any(edges .< 0) && any(edges .> 0) && return false
    det = sum(edges)
    det ≈ 0 && return false
    # Compute scaled hit distance to triangle.
    shear_z = shear[3]
    t_scaled = (
        edges[1] * t_vs[1][3] * shear_z
        + edges[2] * t_vs[2][3] * shear_z
        + edges[3] * t_vs[3][3] * shear_z
    )
    # Test against t_max range.
    det < 0 && (t_scaled >= 0 || t_scaled < ray.t_max * det) && return false
    det > 0 && (t_scaled <= 0 || t_scaled > ray.t_max * det) && return false
    # TODO test against alpha texture if present.
    true
end
