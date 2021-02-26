struct TriangleMesh
    n_triangles::UInt32
    n_vertices::UInt32
    vertices::Vector{Point3f0}
    # For the i-th triangle, its 3 vertex positions are:
    # [vertices[indices[3 * i + j]] for j in 0:2].
    indices::Vector{UInt32}
    # Optional normal vectors, one per vertex.
    normals::Maybe{Vector{Normal3f0}}
    # Optional tangent vectors, one per vertex.
    tangents::Maybe{Vector{Vec3f0}}
    # Optional parametric (u, v) values, one for each vertex.
    uv::Maybe{Vector{Point2f0}}

    function TriangleMesh(
        object_to_world::Transformation,
        n_triangles::Integer, indices::Vector{UInt32},
        n_vertices::Integer, vertices::Vector{Point3f0},
        normals::Maybe{Vector{Normal3f0}} = nothing,
        tangents::Maybe{Vector{Vec3f0}} = nothing,
        uv::Maybe{Vector{Point2f0}} = nothing,
    )
        vertices = object_to_world.(vertices)
        new(
            n_triangles, n_vertices, vertices,
            indices |> maybe_copy, normals |> maybe_copy,
            tangents |> maybe_copy, uv |> maybe_copy,
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
    function Triangle(core::ShapeCore, mesh::TriangleMesh, i::UInt32)
        new(core, mesh, i * 3 + 1)
    end
end

function create_triangle_mesh(
    core::ShapeCore,
    n_triangles::Integer, indices::Vector{UInt32},
    n_vertices::Integer, vertices::Vector{Point3f0},
    normals::Maybe{Vector{Normal3f0}} = nothing,
    tangents::Maybe{Vector{Vec3f0}} = nothing,
    uv::Maybe{Vector{Point2f0}} = nothing,
)
    mesh = TriangleMesh(
        core.object_to_world, n_triangles, indices, n_vertices, vertices,
        normals, tangents, uv,
    )
    [Triangle(core, mesh, i) for i in UnitRange{UInt32}(0:n_triangles - 1)]
end

function area(t::Triangle)
    vs = t |> vertices
    0.5f0 * norm((vs[2] - vs[1]) × (vs[3] - vs[1]))
end

function is_degenerate(vs::AbstractVector{Point3f0})::Bool
    v = (vs[3] - vs[1]) × (vs[2] - vs[1])
    (v ⋅ v) ≈ 0f0
end

function vertices(t::Triangle)
    @SVector Point3f0[t.mesh.vertices[t.mesh.indices[t.i + j]] for j in 0:2]
end
function normals(t::Triangle)
    @SVector Normal3f0[t.mesh.normals[t.mesh.indices[t.i + j]] for j in 0:2]
end
function tangents(t::Triangle)
    @SVector Vec3f0[t.mesh.tangents[t.mesh.indices[t.i + j]] for j in 0:2]
end
function uvs(t::Triangle)
    t.mesh.uv isa Nothing &&
        return @SVector [Point2f0(0), Point2f0(1, 0), Point2f0(1, 1)]
    @SVector [t.mesh.uv[t.i + j] for j in 0:2]
end

function _edge_function(vs)
    Point3f0(
        vs[2][1] * vs[3][2] - vs[2][2] * vs[3][1],
        vs[3][1] * vs[1][2] - vs[3][2] * vs[1][1],
        vs[1][1] * vs[2][2] - vs[1][2] * vs[2][1],
    )
end

object_bound(t::Triangle) = mapreduce(
    v -> v |> t.core.world_to_object |> Bounds3,
    ∪, t |> vertices,
)
world_bound(t::Triangle) = reduce(∪, Bounds3.(t |> vertices))

function _to_ray_coordinate_space(
    vertices::AbstractVector{Point3f0}, ray::Union{Ray, RayDifferentials}
)
    # Compute permutation.
    kz = ray.d .|> abs |> argmax
    kx = kz + 1
    kx == 4 && (kx = 1)
    ky = kx + 1
    ky == 4 && (ky = 1)
    permutation = @SVector [kx, ky, kz]
    # Permute ray direction.
    d = ray.d[permutation]
    # Compute shear.
    denom = 1f0 / d[3]
    shear = Point3f0(-d[1] * denom, -d[2] * denom, denom)
    # Translate, apply permutation and shear to vertices.
    tvs = @SVector Point3f0[
        (vertices[i] - ray.o)[permutation] + Point3f0(
            shear[1] * (vertices[i][kz] - ray.o[kz]),
            shear[2] * (vertices[i][kz] - ray.o[kz]),
            0f0,
        ) for i in 1:3
    ]
    tvs, shear
end

function ∂p(
    t::Triangle, vs::AbstractVector{Point3f0}, uv::AbstractVector{Point2f0},
)::Tuple{Vec3f0, Vec3f0, Vec3f0, Vec3f0}
    # Compute deltas for partial derivative matrix.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δp_13, δp_23 = Vec3f0(vs[1] - vs[3]), Vec3f0(vs[2] - vs[3])
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    if det ≈ 0
        v = normalize((vs[3] - vs[1]) × (vs[2] - vs[1]))
        _, ∂p∂u, ∂p∂v = coordinate_system(v, Vec3f0(0f0))
        return ∂p∂u, ∂p∂v, δp_13, δp_23
    end
    inv_det = 1f0 / det
    ∂p∂u = Vec3f0( δuv_23[2] * δp_13 - δuv_13[2] * δp_23) * inv_det
    ∂p∂v = Vec3f0(-δuv_23[1] * δp_13 + δuv_13[1] * δp_23) * inv_det
    ∂p∂u, ∂p∂v, δp_13, δp_23
end

function ∂n(
    t::Triangle, uv::AbstractVector{Point2f0},
)::Tuple{Normal3f0, Normal3f0}
    t.mesh.normals isa Nothing && return Normal3f0(0), Normal3f0(0)
    t_normals = t |> normals
    # Compute deltas for partial detivatives of normal.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δn_13, δn_23 = t_normals[1] - t_normals[3], t_normals[2] - t_normals[3]
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    det ≈ 0 && return Normal3f0(0), Normal3f0(0)

    inv_det = 1f0 / det
    ∂n∂u = ( δuv_23[2] * δn_13 - δuv_13[2] * δn_23) * inv_det
    ∂n∂v = (-δuv_23[1] * δn_13 + δuv_13[1] * δn_23) * inv_det
    ∂n∂u, ∂n∂v
end

function _init_triangle_shading_geometry!(
    t::Triangle, interaction::SurfaceInteraction{Triangle},
    barycentric::Point3f0, uv::AbstractVector{Point2f0},
)
    !(t.mesh.normals ≢ nothing || t.mesh.tangents ≢ nothing) && return
    # Initialize triangle shading geometry.
    # Compute shading normal, tangent & bitangent.
    ns = interaction.core.n
    if t.mesh.normals ≢ nothing
        ns = sum_mul(barycentric, normals(t)) |> normalize
    end
    if t.mesh.tangents ≢ nothing
        ss = sum_mul(barycentric, tangents(t)) |> normalize
    else
        ss = interaction.∂p∂u |> normalize
    end
    ts = ns × ss
    if (ts ⋅ ts) > 0
        ts = ts |> normalize |> Vec3f0
        ss = Vec3f0(ts × ns)
    else
        _, ss, ts = coordinate_system(ns, ss)
    end
    ∂n∂u, ∂n∂v = ∂n(t, uv)
    set_shading_geometry!(interaction, ss, ts, ∂n∂u, ∂n∂v, true)
end

function intersect(
    t::Triangle, ray::Union{Ray, RayDifferentials},
    test_alpha_texture::Bool = false,
)::Tuple{Bool, Maybe{Float32}, Maybe{SurfaceInteraction}}
    vs = t |> vertices
    is_degenerate(vs) && return false, nothing, nothing
    t_vs, shear = _to_ray_coordinate_space(vs, ray)
    # Compute edge function coefficients.
    edges = t_vs |> _edge_function
    if iszero(edges) # Fall-back to double precision.
        edges = t_vs .|> (x -> x .|> Float64) |> _edge_function
    end
    # Perform triangle edge & determinant tests.
    # Point is inside a triangle if all edges have the same sign.
    any(edges .< 0) && any(edges .> 0) && return false, nothing, nothing
    det = edges |> sum
    det ≈ 0 && return false, nothing, nothing
    # Compute scaled hit distance to triangle.
    shear_z = shear[3]
    t_scaled = (
        edges[1] * t_vs[1][3] * shear_z
        + edges[2] * t_vs[2][3] * shear_z
        + edges[3] * t_vs[3][3] * shear_z
    )
    # Test against t_max range.
    det < 0 && (t_scaled >= 0 || t_scaled < ray.t_max * det) &&
        return false, nothing, nothing
    det > 0 && (t_scaled <= 0 || t_scaled > ray.t_max * det) &&
        return false, nothing, nothing
    # Compute barycentric coordinates and t value for triangle intersection.
    inv_det = 1f0 / det
    barycentric = edges .* inv_det
    t_hit = t_scaled * inv_det
    # TODO check that t_hit > 0
    uv = t |> uvs
    ∂p∂u, ∂p∂v, δp_13, δp_23 = ∂p(t, vs, uv)
    # Interpolate (u, v) paramteric coordinates and hit point.
    hit_point = sum_mul(barycentric, vs)
    uv_hit = sum_mul(barycentric, uv)

    interaction = SurfaceInteraction(
        hit_point, ray.time, -ray.d, uv_hit,
        ∂p∂u, ∂p∂v, Normal3f0(0), Normal3f0(0), t,
    )
    interaction.core.n = interaction.shading.n = normalize(δp_13 × δp_23)
    t.mesh isa Nothing && return true, t_hit, interaction
    _init_triangle_shading_geometry!(t, interaction, barycentric, uv)
    # Ensure correct orientation of the geometric normal.
    if t.mesh.normals ≢ nothing
        interaction.core.n = face_forward(
            interaction.core.n, interaction.shading.n,
        )
    elseif t.core.reverse_orientation ⊻ t.core.transform_swaps_handedness
        interaction.core.n = interaction.shading.n = -interaction.core.n
    end
    # TODO test against alpha texture if present.
    true, t_hit, interaction
end

function intersect_p(
    t::Triangle, ray::Union{Ray, RayDifferentials},
    test_alpha_texture::Bool = false,
)::Bool
    vs = t |> vertices
    is_degenerate(vs) && return false, nothing, nothing
    t_vs, shear = _to_ray_coordinate_space(vs, ray)
    # Compute edge function coefficients.
    edges = t_vs |> _edge_function
    if iszero(edges) # Fall-back to double precision.
        edges = t_vs .|> (x -> x .|> Float64) |> _edge_function
    end
    # Perform triangle edge & determinant tests.
    # Point is inside a triangle if all edges have the same sign.
    any(edges .< 0) && any(edges .> 0) && return false
    det = edges |> sum
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
