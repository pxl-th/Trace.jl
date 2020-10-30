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
        # Transform vertices to world space.
        # vertices = [object_to_world(v) for v in vertices]
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

function is_degenerate(t::Triangle)::Bool
    vs = t |> vertices
    v = (vs[3] - vs[1]) × (vs[2] - vs[1])
    (v ⋅ v) ≈ 0 ? true : false
end

vertices(t::Triangle) = [t.mesh.vertices[t.mesh.indices[t.i + j]] for j in 0:2]
normals(t::Triangle) = [t.mesh.normals[t.mesh.indices[t.i + j]] for j in 0:2]
tangents(t::Triangle) = [t.mesh.tangents[t.mesh.indices[t.i + j]] for j in 0:2]
function uvs(t::Triangle)
    t.mesh.uv isa Nothing && return [Point2f0(0), Point2f0(1, 0), Point2f0(1, 1)]
    [t.mesh.uv[t.i + j] for j in 0:2]
end

object_bound(t::Triangle) = mapreduce(v -> v |> t.core.world_to_object |> Bounds3, ∪, t |> vertices)
world_bound(t::Triangle) = reduce(∪, Bounds3.(t |> vertices))

function _edge_function(vs::Vector{Point3{T}}) where T <: Union{Float32, Float64}
    Point3f0(
        vs[2][1] * vs[3][2] - vs[2][2] * vs[3][1],
        vs[3][1] * vs[1][2] - vs[3][2] * vs[1][1],
        vs[1][1] * vs[2][2] - vs[1][2] * vs[2][1],
    )
end

function _to_ray_coordinate_space(vs::Vector{Point3f0}, ray::Ray)
    # Transform vs.
    vs .-= ray.o
    # Permute vs & ray direction.
    kz = ray.d .|> abs |> argmax
    kx = kz + 1
    kx == 4 && (kx = 1)
    ky = kx + 1
    ky == 4 && (ky = 1)

    d = ray.d[[kx, ky, kz]]
    vs = [v[[kx, ky, kz]] for v in vs]
    # Apply shear transformation to vs.
    shear = Point3f0(-d[1] / d[3], -d[2] / d[3], 1f0 / d[3])
    Point3f0[v + Point3f0(shear[1] * v[3], shear[2] * v[3], 0f0) for v in vs], shear
end

function ∂p(
    t::Triangle, vs::Vector{Point3f0}, uv::Vector{Point2f0},
)::Tuple{Vec3f0, Vec3f0, Vec3f0, Vec3f0}
    # Compute deltas for partial derivative matrix.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δp_13, δp_23 = Vec3f0(vs[1] - vs[3]), Vec3f0(vs[2] - vs[3])
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    if det ≈ 0
        v = normalize((vs[3] - vs[1]) × (vs[2] - vs[1]))
        return coordinate_system(v, Vec3f0(0))[2:3], δp_13, δp_23
    end
    inv_det = 1f0 / det
    ∂p∂u = Vec3f0( δuv_23[2] * δp_13 - δuv_13[2] * δp_23) * inv_det
    ∂p∂v = Vec3f0(-δuv_23[1] * δp_13 + δuv_13[1] * δp_23) * inv_det
    ∂p∂u, ∂p∂v, δp_13, δp_23
end

function ∂n(t::Triangle, uv::Vector{Point2f0})::Tuple{Normal3f0, Normal3f0}
    t.mesh.normals isa Nothing && return Normal3f0(0), Normal3f0(0)
    normals = [t.mesh.normals[t.i + j] for j in 0:2]
    # Compute deltas for partial detivatives of normal.
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δn_13, δn_23 = normals[1] - normals[3], normals[2] - normals[3]
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    det ≈ 0 && return Normal3f0(0), Normal3f0(0)

    inv_det = 1f0 / det
    ∂n∂u = ( δuv_23[2] * δn_13 - δuv_13[2] * δn_23) * inv_det
    ∂n∂v = (-δuv_23[1] * δn_13 + δuv_13[1] * δn_23) * inv_det
    ∂n∂u, ∂n∂v
end

function _init_triangle_shading_geometry!(
    t::Triangle, interaction::SurfaceInteraction{Triangle},
    barycentric::Point3f0, uv::Vector{Point2f0},
)
    # Initialize triangle shading geometry.
    if t.mesh.normals ≢ nothing || t.mesh.tangents ≢ nothing
        # Compute shading normal, tangent & bitangent.
        ns = interaction.core.n
        if t.mesh.normals ≢ nothing
            ns = normalize(sum_mul(barycentric, normals(t)))
        end
        if t.mesh.tangents ≢ nothing
            ss = normalize(sum_mul(barycentric, tangents(t)))
        else
            ss = interaction.∂p∂u |> normalize
        end
        ts = ns × ss
        if (ts ⋅ ts) > 0
            ts = ts |> normalize
            ss = ts × ns
        else
            ss, ts = coordinate_system(ns, ss)[2:3]
        end
        ∂n∂u, ∂n∂v = ∂n(t, uv)
        set_shading_geometry!(interaction, Vec3f0(ss), Vec3f0(ts), ∂n∂u, ∂n∂v, true)
    end
end

function intersect(
    t::Triangle, ray::Ray, test_alpha_texture::Bool = false,
)::Tuple{Bool, Maybe{Float32}, Maybe{SurfaceInteraction}}
    is_degenerate(t) && return false, nothing, nothing
    vs = t |> vertices
    t_vs, shear = _to_ray_coordinate_space(vs, ray)
    # Compute edge function coefficients.
    edges = t_vs |> _edge_function
    if all(edges .≈ 0) # Fall-back to double precision.
        edges = t_vs .|> (x -> x .|> Float64) |> _edge_function
    end
    # Perform triangle edge & determinant tests.
    # Point is inside a triangle if all edges have the same sign.
    any(edges .< 0) && any(edges .> 0) && return false, nothing, nothing
    det = edges |> sum
    det ≈ 0 && return false, nothing, nothing
    # Compute scaled hit distance to triangle.
    t_vs .*= Point3f0(1f0, 1f0, shear[3])
    t_scaled = edges[1] * t_vs[1][3] + edges[2] * t_vs[2][3] + edges[3] * t_vs[3][3]
    # Test against t_max range.
    det < 0 && (t_scaled >= 0 || t_scaled < ray.t_max * det) && return false, nothing, nothing
    det > 0 && (t_scaled <= 0 || t_scaled > ray.t_max * det) && return false, nothing, nothing
    # Compute barycentric coordinates and t value for triangle intersection.
    inv_det = 1f0 / det
    barycentric = edges .* inv_det
    t_hit = t_scaled * inv_det
    # TODO check that t > 0 w.r.t. error bounds.
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
        interaction.core.n = face_forward(interaction.core.n, interaction.shading.n)
    elseif t.core.reverse_orientation ⊻ t.core.transform_swaps_handedness
        interaction.core.n = interaction.shading.n = -interaction.core.n
    end
    # TODO test against alpha texture if present (when we have textures).
    true, t_hit, interaction
end
