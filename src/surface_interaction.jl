mutable struct Interaction
    """
    Intersection point in world coordinates.
    """
    p::Point3f0
    """
    Time of intersection.
    """
    time::Float32
    """
    Negative direction of ray (for ray-shape interactions)
    in world coordinates.
    """
    wo::Vec3f0
    """
    Surface normal at the point in world coordinates.
    """
    n::Normal3f0
end

mutable struct ShadingInteraction
    n::Normal3f0
    ∂p∂u::Vec3f0
    ∂p∂v::Vec3f0
    ∂n∂u::Normal3f0
    ∂n∂v::Normal3f0
end

mutable struct SurfaceInteraction{S <: AbstractShape}
    core::Interaction
    shading::ShadingInteraction
    uv::Point2f0

    ∂p∂u::Vec3f0
    ∂p∂v::Vec3f0
    ∂n∂u::Normal3f0
    ∂n∂v::Normal3f0

    shape::Maybe{S}
    primitive::Maybe{P} where P <: Primitive
    bsdf # TODO ::Maybe{BSDF}

    ∂u∂x::Float32
    ∂u∂y::Float32
    ∂v∂x::Float32
    ∂v∂y::Float32
    ∂p∂x::Vec3f0
    ∂p∂y::Vec3f0
end

function SurfaceInteraction(
    p::Point3f0, time::Float32, wo::Vec3f0, uv::Point2f0,
    ∂p∂u::Vec3f0, ∂p∂v::Vec3f0, ∂n∂u::Normal3f0, ∂n∂v::Normal3f0,
    shape::Maybe{S} = nothing, primitive::Maybe{P} = nothing,
) where S <: AbstractShape where P <: Primitive
    n = (∂p∂u × ∂p∂v) |> normalize
    if !(shape isa Nothing) && (shape.core.reverse_orientation ⊻ shape.core.transform_swaps_handedness)
        n *= -1
    end

    core = Interaction(p, time, wo, n)
    shading = ShadingInteraction(n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
    SurfaceInteraction{typeof(shape)}(
        core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
        shape, primitive, nothing,
        0f0, 0f0, 0f0, 0f0, Vec3f0(0f0), Vec3f0(0f0),
    )
end

function set_shading_geometry!(
    i::SurfaceInteraction, tangent::Vec3f0, bitangent::Vec3f0,
    ∂n∂u::Normal3f0, ∂n∂v::Normal3f0, orientation_is_authoritative::Bool,
)
    i.shading.n = normalize(tangent × bitangent)
    if !(i.shape isa Nothing) && (i.shape.core.reverse_orientation ⊻ i.shape.core.transform_swaps_handedness)
        i.shading.n *= -1
    end
    if orientation_is_authoritative
        i.core.n = face_forward(i.core.n, i.shading.n)
    else
        i.shading.n = face_forward(i.shading.n, i.core.n)
    end

    i.shading.∂p∂u = tangent
    i.shading.∂p∂v = bitangent
    i.shading.∂n∂u = ∂n∂u
    i.shading.∂n∂v = ∂n∂v
end

is_surface_interaction(i::Interaction) = i.n != Normal3f0(0)

"""
Compute partial derivatives needed for computing sampling rates
for things like texture antialiasing.
"""
function compute_differentials!(si::SurfaceInteraction, ray::RayDifferentials)
    if !ray.has_differentials
        si.∂u∂x = si.∂v∂x = 0f0
        si.∂u∂y = si.∂v∂y = 0f0
        si.∂p∂x = si.∂p∂y = Vec3f0(0f0)
        return
    end
    # Estimate screen change in p and (u, v).
    # Compute auxiliary intersection points with plane.
    d = -(si.core.n ⋅ si.core.p)
    tx = (-(si.core.n ⋅ ray.rx_origin) - d) / (si.core.n ⋅ ray.rx_direction)
    ty = (-(si.core.n ⋅ ray.ry_origin) - d) / (si.core.n ⋅ ray.ry_direction)
    px = ray.rx_origin + tx * ray.rx_direction
    py = ray.ry_origin + ty * ray.ry_direction

    si.∂p∂x = px - si.core.p
    si.∂p∂y = py - si.core.p
    # Compute (u, v) offsets at auxiliary points.
    # Choose two dimensions for ray offset computation.
    n = si.core.n .|> abs
    if n[1] > n[2] && n[1] > n[3]
        dim = Point2(2, 3)
    elseif n[2] > n[3]
        dim = Point2(1, 3)
    else
        dim = Point2(1, 2)
    end
    # Initialization for offset computation.
    a = Mat2f0(dim[1], dim[1], dim[2], dim[2])
    bx = Point2f0(px[dim[1]] - si.core.p[dim[1]], px[dim[2]] - si.core.p[dim[2]])
    by = Point2f0(py[dim[1]] - si.core.p[dim[1]], py[dim[2]] - si.core.p[dim[2]])
    sx = a \ bx
    sy = a \ by

    si.∂u∂x, si.∂v∂x = any(isnan.(sx)) ? (0f0, 0f0) : sx
    si.∂u∂y, si.∂v∂y = any(isnan.(sy)) ? (0f0, 0f0) : sy
end

"""
If an intersection was found, it is necessary to determine, how
the surface's material scatters light.
`compute_scattering!` method evaluates texture functions to determine
surface properties and then initializing a representation of the BSDF
at the point.
"""
function compute_scattering!(
    si::SurfaceInteraction, ray::RayDifferentials,
    allow_multiple_lobes::Bool = false, ::Type{T} = Radiance,
) where T <: TransportMode
    compute_differentials!(si, ray)
    compute_scattering!(si.primitive, si, allow_multiple_lobes, T)
end

@inline function le(si::SurfaceInteraction, w::Vec3f0)::RGBSpectrum
    # TODO right now return 0, since there is no area lights implemented.
    RGBSpectrum(0f0)
end

function (t::Transformation)(sc::Interaction)
    Interaction(
        sc.p |> t,
        sc.time,
        sc.wo |> t |> normalize,
        sc.n |> t |> normalize,
    )
end
function (t::Transformation)(sh::ShadingInteraction)
    ShadingInteraction(
        sh.n |> t |> normalize,
        sh.∂p∂u |> t, sh.∂p∂v |> t,
        sh.∂n∂u |> t, sh.∂n∂v |> t,
    )
end
function (t::Transformation)(si::SurfaceInteraction)
    # TODO compute shading normal separately
    core = si.core |> t
    shading = si.shading |> t
    SurfaceInteraction(
        core, shading, si.uv,
        si.∂p∂u |> t, si.∂p∂v |> t,
        si.∂n∂u |> t, si.∂n∂v |> t,
        si.shape, si.primitive, si.bsdf,
        si.∂u∂x, si.∂u∂y, si.∂v∂x, si.∂v∂y,
        si.∂p∂x |> t, si.∂p∂y |> t,
    )
end
