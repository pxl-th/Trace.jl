module Trace

import FileIO
using Parameters: @with_kw
using LinearAlgebra
using StaticArrays
using GeometryBasics

GeometryBasics.@fixed_vector Normal StaticVector
const Normal3f0 = Normal{3, Float32}

Maybe{T} = Union{T, Nothing}
maybe_copy(v::Maybe)::Maybe = v isa Nothing ? v : copy(v)

abstract type AbstractRay end
abstract type Spectrum end
abstract type AbstractShape end
abstract type Primitive end
abstract type Light end
abstract type Material end
abstract type BxDF end

const Radiance = Val{:Radiance}
const Importance = Val{:Importance}
const TransportMode = Union{Radiance, Importance}

function concentric_sample_disk(u::Point2f0)::Point2f0
    # Map uniform random numbers to [-1, 1].
    offset = 2f0 * u - Vec2f0(1f0)
    # Handle degeneracy at the origin.
    offset[1] ≈ 0 && offset[2] ≈ 0 && return Point2f0(0)
    if abs(offset[1]) > abs(offset[2])
        r = offset[1]
        θ = (offset[2] / offset[1]) * π / 4
    else
        r = offset[2]
        θ = π / 2 - (offset[1] / offset[2]) * π / 4
    end
    r * Point2f0(θ |> cos, θ |> sin)
end

@inline function cosine_sample_hemisphere(u::Point2f0)::Vec3f0
    d = u |> concentric_sample_disk
    z = √max(0f0, 1f0 - d[1] ^ 2 - d[2] ^ 2)
    Vec3f0(d[1], d[2], z)
end

@inline sum_mul(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

"""
Since normal is (0, 0, 1), cos_θ between n & w is (0, 0, 1) ⋅ w = w.z.
"""
@inline cos_θ(w::Vec3f0) = w[3]
@inline sin_θ2(w::Vec3f0) = max(0f0, 1f0 - cos_θ(w) * cos_θ(w))
@inline sin_θ(w::Vec3f0) = w |> sin_θ2 |> √

@inline function cos_ϕ(w::Vec3f0)
    sinθ = w |> sin_θ
    sinθ ≈ 0f0 ? 1f0 : clamp(w[1] / sinθ, -1f0, 1f0)
end
@inline function sin_ϕ(w::Vec3f0)
    sinθ = w |> sin_θ
    sinθ ≈ 0f0 ? 1f0 : clamp(w[2] / sinθ, -1f0, 1f0)
end

"""
Reflect `wo` about `n`.
"""
@inline reflect(wo::Vec3f0, n::Vec3f0) = -wo + 2f0 * (wo ⋅ n) * n

function find_interval(size::Int64, predicate::Function)
    first, len = 0, size
    while len > 1
        half = len >> 1
        middle = first + half
        if predicate(middle)
            first = middle + 1
            len -= half + 1
        else
            len = half
        end
    end
    clamp(first, 1, size - 1)
end

function partition!(x::Vector, range::UnitRange, predicate::Function)
    left = range[1]
    for i in range
        if left != i && predicate(x[i])
            x[i], x[left] = x[left], x[i]
            left += 1
        end
    end
    left
end

function coordinate_system(v1::Vec3f0, v2::Vec3f0)
    if abs(v1[1]) > abs(v1[2])
        v2 = typeof(v2)(-v1[3], 0, v1[1]) / sqrt(v1[1] * v1[1] + v1[3] * v1[3])
    else
        v2 = typeof(v2)(0, v1[3], -v1[2]) / sqrt(v1[2] * v1[2] + v1[3] * v1[3])
    end
    v1, v2, v1 × v2
end

function spherical_direction(sin_θ::Float32, cos_θ::Float32, ϕ::Float32)
    Vec3f0(sin_θ * cos(ϕ), sin_θ * sin(ϕ), cos_θ)
end
function spherical_direction(
    sin_θ::Float32, cos_θ::Float32, ϕ::Float32,
    x::Vec3f0, y::Vec3f0, z::Vec3f0,
)
    sin_θ * cos(ϕ) * x + sin_θ * sin(ϕ) * y + cos_θ * z
end

spherical_θ(v::Vec3f0) = clamp(v[3], -1, 1) |> acos
function spherical_ϕ(v::Vec3f0)
    p = atan(v[2], v[1])
    p < 0 ? p + 2 * π : p
end


"""
Flip normal `n` so that it lies in the same hemisphere as `v`.
"""
face_forward(n, v) = (n ⋅ v) < 0 ? -n : n

include("ray.jl")
include("bounds.jl")
include("transformations.jl")
include("spectrum.jl")

include("surface_interaction.jl")

# TODO AnimatedTransform, AnimatedBounds
# TODO Medium & add it to structs


struct Scene
    lights::Vector{L} where L <: Light
    aggregate::P where P <: Primitive
    bound::Bounds3

    function Scene(
        lights::Vector{L}, aggregate::P,
    ) where L <: Light where P <: Primitive
        # TODO preprocess for lights
        new(lights, aggregate, aggregate |> world_bound)
    end
end

@inline function intersect!(scene::Scene, ray::Union{Ray, RayDifferentials})
    intersect!(scene.aggregate, ray)
end
@inline function intersect_p(scene::Scene, ray::Union{Ray, RayDifferentials})
    intersect_p(scene.aggregate, ray)
end

@inline function spawn_ray(p0::Interaction, p1::Interaction)::Ray
    Ray(o=p0.p, d=p1.p - p0.p, time=p0.time)
end
@inline function spawn_ray(p0::SurfaceInteraction, p1::Interaction)::Ray
    spawn_ray(p0.core, p1)
end
@inline function spawn_ray(si::SurfaceInteraction, direction::Vec3f0)::Ray
    Ray(o=si.core.p, d=direction, time=si.core.time)
end

include("shapes/Shape.jl")
include("primitive.jl")
include("accel/bvh.jl")

include("filter.jl")
include("film.jl")
include("reflection/Reflection.jl")

include("camera/camera.jl")
include("sampler/sampler.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("materials/bsdf.jl")
include("materials/material.jl")

include("lights/emission.jl")
include("lights/light.jl")
include("lights/point.jl")
include("lights/directional.jl")

include("integrators/sampler.jl")

# filter = LanczosSincFilter(Point2f0(4f0), 3f0)
# width, height = 28f0, 28f0
# film = Film(
#     Point2f0(width, height), Bounds2(Point2f0(0f0), Point2f0(1f0)),
#     filter, 35f0, 1f0, "output.png",
# )
# frame = width / height
# screen = Bounds2(Point2f0(-frame, -1f0), Point2f0(frame, 1f0))
# camera = PerspectiveCamera(
#     translate(Vec3f0(0)), screen,
#     0f0, 1f0, 0f0, 1f6, 35f0, film,
# )
# sampler = UniformSampler(1)
# whitted = WhittedIntegrator(camera, sampler, 1)

# light = PointLight(translate(Vec3f0(0, 1.5, -100)), RGBSpectrum(1f0))
# core = ShapeCore(translate(Vec3f0(0, 0, -100)), false)
# matte = MatteMaterial(
#     ConstantTexture(RGBSpectrum(1f0, 0f0, 0f0)),
#     ConstantTexture(1f0),
# )
# sphere = GeometricPrimitive(Sphere(core, 1f0, -1f0, 1f0, 360f0), matte)
# scene = Scene([light], sphere)

# whitted(scene)


material = MatteMaterial(
    ConstantTexture(RGBSpectrum(1f0, 0.2f0, 0.1f0)),
    ConstantTexture(0f0),
)
# core = ShapeCore(Transformation(), true)
core = ShapeCore(translate(Vec3f0(0.0125f0, 0.0125f0, -3f0)), false)
sphere = Sphere(core, 0.01f0, -1f0, 1f0, 360f0)
primitive = GeometricPrimitive(sphere, material)
bvh = BVHAccel{SAH}([primitive])

lights = [PointLight(
    translate(Vec3f0(0f0, 0f0, 0f0)), RGBSpectrum(Float32(3 * π)),
)]
scene = Scene(lights, bvh)
# Construct Film and Camera.
resolution = Point2f0(128f0, 128f0)
filter = LanczosSincFilter(Point2f0(1f0), 3f0)
film = Film(
    resolution, Bounds2(Point2f0(0f0), Point2f0(1f0)),
    filter, 1f0, 1f0, "test-output-2.png",
)
screen = Bounds2(Point2f0(-1f0), Point2f0(1f0))
camera = PerspectiveCamera(
    Transformation(), screen, 0f0, 1f0, 0f0, 10f0, 45f0, film,
)

sampler = UniformSampler(1)
integrator = WhittedIntegrator(camera, sampler, 2)
scene |> integrator

"""
TODO
- assert that t_max in intersect methods >= 0
- test if bvh contains duplicates
"""

# @info from_RGB(SampledSpectrum, Point3f0(1f0, 0f0, 0f0), Illuminant)
# @info from_RGB(SampledSpectrum, Point3f0(1f0, 0f0, 0f0), Reflectance)
# @info from_XYZ(SampledSpectrum, Point3f0(0.5f0, 0f0, 0.5f0))
# @info from_RGB(RGBSpectrum, Point3f0(0.5f0, 0f0, 0.5f0))
# @info from_XYZ(RGBSpectrum, Point3f0(0.5f0, 0f0, 0.5f0))

# @info Point3f0(1.0, 0.0, 0.0) |> RGB_to_XYZ
# @info Point3f0(1.0, 0.0, 0.0) |> RGB_to_XYZ |> XYZ_to_RGB

end
