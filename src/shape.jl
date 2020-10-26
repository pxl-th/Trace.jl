abstract type AbstractShape end

struct ShapeCore
    object_to_world::Transformation
    world_to_object::Transformation
    reverse_orientation::Bool
    transform_swaps_handedness::Bool

    function ShapeCore(
        object_to_world::Transformation, world_to_object::Transformation,
        reverse_orientation::Bool,
    )
        new(
            object_to_world, world_to_object, reverse_orientation,
            object_to_world |> swaps_handedness,
        )
    end
end

# Interface.
function area(::AbstractShape)::Float32 end
function object_bound(::AbstractShape)::Bounds3 end
function world_bound(s::AbstractShape)::Bounds3
    s |> object_bound |> s.core.object_to_world
end
# function intersect(
#     ::AbstractShape, ::Ray, test_alpha_texture::Bool = true,
# )::Tuple{Float32, SurfaceInteraction} end

struct Sphere <: AbstractShape
    core::ShapeCore

    radius::Float32
    # Implicit constraints.
    z_min::Float32
    z_max::Float32
    # Parametric constraints.
    θ_min::Float32
    θ_max::Float32
    ϕ_max::Float32

    function Sphere(
        core::ShapeCore, radius::Float32,
        z_min::Float32, z_max::Float32, ϕ_max::Float32,
    )
        new(
            core, radius,
            clamp(min(z_min, z_max), -radius, radius),
            clamp(max(z_min, z_max), -radius, radius),
            acos(clamp(z_min / radius, -1f0, 1f0)),
            acos(clamp(z_max / radius, -1f0, 1f0)),
            deg2rad(clamp(ϕ_max, 0f0, 360f0)),
        )
    end
end

function object_bound(s::Sphere)
    Bounds3(
        Point3f0(-s.radius, -s.radius, s.z_min),
        Point3f0(s.radius, s.radius, s.z_max),
    )
end

function solve_quadratic(a::Float32, b::Float32, c::Float32)
    # Find disriminant.
    d = b * b - 4 * a * c
    if d < 0
        return false, NaN32, NaN32
    end
    d = d |> sqrt
    # Compute roots.
    q = -0.5f0 * (b + (b < 0 ? -d : d))
    t0 = q / a
    t1 = c / q
    if t0 > t1
        t0, t1 = t1, t0
    end
    true, t0, t1
end

function refine_intersection(p::Point, s::Sphere)
    p *= s.radius ./ distance(Point3f0(0), p)
    p[1] ≈ 0 && p[2] ≈ 0 && (p *= Point3f0(1e-5, 0f0, 0f0))
    p
end

"""
Test if hit point exceeds clipping parameters of the sphere.
"""
function test_clipping(s::Sphere, p::Point3f0, ϕ::Float32)::Bool
    (s.z_min > -s.radius && p[3] < s.z_min) ||
    (s.z_max < s.radius && p[3] > s.z_max) ||
    ϕ > s.ϕ_max
end

function compute_ϕ(p::Point3f0)
    ϕ = atan(p[2], p[1])
    ϕ < 0f0 && (ϕ += 2f0 * π)
    ϕ
end

function precompute_ϕ(p::Point3f0)
    z_radius = sqrt(p[1] * p[1] + p[2] * p[2])
    inv_z_radius = 1f0 / z_radius
    cos_ϕ = p[1] * inv_z_radius
    sin_ϕ = p[2] * inv_z_radius
    sin_ϕ, cos_ϕ
end

"""
Compute partial derivatives of intersection point in parametric form.
"""
function ∂p(s::Sphere, p::Point3f0, θ::Float32, sin_ϕ::Float32, cos_ϕ::Float32)
    ∂p∂u = Vec3f0(-s.ϕ_max * p[2], s.ϕ_max * p[1], 0f0)
    ∂p∂v = (s.θ_max - s.θ_min) * Vec3f0(
        p[3] * cos_ϕ, p[3] * sin_ϕ, -s.radius * sin(θ),
    )
    ∂p∂u, ∂p∂v, sin_ϕ, cos_ϕ
end

function ∂n(
    s::Sphere, p::Point3f0, θ::Float32,
    ∂p∂u::Vec3f0, ∂p∂v::Vec3f0,
    sin_ϕ::Float32, cos_ϕ::Float32,
)
    ∂2p∂u2 = -s.ϕ_max * s.ϕ_max * Vec3f0(p[1], p[2], 0f0)
    ∂2p∂u∂v = (s.θ_max - s.θ_min) * p[3] * s.ϕ_max * Vec3f0(-sin_ϕ, cos_ϕ, 0f0)
    ∂2p∂v2 = (s.θ_max - s.θ_min) ^ 2 * -p
    # Compute coefficients for fundamental forms.
    E = ∂p∂u ⋅ ∂p∂u
    F = ∂p∂u ⋅ ∂p∂v
    G = ∂p∂v ⋅ ∂p∂v
    n = normalize(∂p∂u × ∂p∂v)
    e = n ⋅ ∂2p∂u2
    f = n ⋅ ∂2p∂u∂v
    g = n ⋅ ∂2p∂v2
    # Compute derivatives from fundamental form coefficients.
    inv_egf = 1f0 / (E * G - F * F)
    ∂n∂u = Normal3f0(
        (f * F - e * G) * inv_egf * ∂p∂u +
        (e * F - f * E) * inv_egf * ∂p∂v
    )
    ∂n∂v = Normal3f0(
        (g * F - f * G) * inv_egf * ∂p∂u +
        (f * F - g * E) * inv_egf * ∂p∂v
    )
    ∂n∂u, ∂n∂v
end

function intersect(s::Sphere, ray::Ray, test_alpha_texture::Bool)
    # Transform ray to object space.
    or::Ray = ray |> s.core.world_to_object
    # Substitute ray into sphere equation.
    a = or.d |> norm
    b = 2 * or.o ⋅ or.d
    c = or.o |> norm - or.r * or.r
    # Solve quadratic equation for t.
    exists, t0, t1 = solve_quadratic(a, b, c)
    !exists && return false
    (t0 > or.t_max || t1 < 0f0) && return false

    hit_point = t0 |> or |> refine_intersection
    ϕ = hit_point |> compute_ϕ
    # Test sphere intersection against clipping parameters.
    if test_clipping(s, hit_point, ϕ)
        hit_point = t1 |> or |> refine_intersection
        ϕ = hit_point |> compute_ϕ
        test_clipping(s, hit_point, ϕ) && return false
    end
    # Find parametric representation of hit point.
    u = ϕ / s.ϕ_max
    θ = clamp(hit_point[3] / s.radius, -1f0, 1f0) |> acos
    v = (θ - s.θ_min) / (s.θ_max - s.θ_min)

    sin_ϕ, cos_ϕ = hit_point |> precompute_ϕ
    ∂p∂u, ∂p∂v = ∂p(s, hit_point, θ, sin_ϕ, cos_ϕ)
    ∂n∂u, ∂n∂v = ∂n(s, hit_point, θ, ∂p∂u, ∂p∂v, sin_ϕ, cos_ϕ)

    true
end
