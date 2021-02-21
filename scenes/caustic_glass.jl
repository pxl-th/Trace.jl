using GeometryBasics
using Trace
using ProgressMeter
using Printf
using FileIO
using ImageCore
using LinearAlgebra

using StatProfilerHTML
using BenchmarkTools
using Profile

function render()
    model = raw"./scenes/models/caustic-glass.ply"

    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(0f0),
        Trace.ConstantTexture(0f0),
        Trace.ConstantTexture(1.2f0),
        true,
    )

    # plastic_red = Trace.PlasticMaterial(
    #     Trace.ConstantTexture(Trace.RGBSpectrum(0.8f0, 0.235f0, 0.2f0)),
    #     Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
    #     Trace.ConstantTexture(0.010408001f0),
    #     true,
    # )
    # plastic_green = Trace.PlasticMaterial(
    #     Trace.ConstantTexture(Trace.RGBSpectrum(0.219f0, 0.596f0, 0.149f0)),
    #     Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
    #     Trace.ConstantTexture(0.010408001f0),
    #     true,
    # )
    # plastic_purple = Trace.PlasticMaterial(
    #     Trace.ConstantTexture(Trace.RGBSpectrum(0.584f0, 0.345f0, 0.698f0)),
    #     Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
    #     Trace.ConstantTexture(0.010408001f0),
    #     true,
    # )
    plastic = Trace.PlasticMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
        Trace.ConstantTexture(0.010408001f0),
        true,
    )

    # sphere_primitive1 = Trace.GeometricPrimitive(Trace.Sphere(
    #     Trace.ShapeCore(Trace.translate(Vec3f0(-1.61, 0.31, -98)), false),
    #     0.3f0, 360f0,
    # ), plastic_red)
    # sphere_primitive2 = Trace.GeometricPrimitive(Trace.Sphere(
    #     Trace.ShapeCore(Trace.translate(Vec3f0(-1.3, 0.31, -98.61)), false),
    #     0.3f0, 360f0,
    # ), plastic_green)
    # sphere_primitive3 = Trace.GeometricPrimitive(Trace.Sphere(
    #     Trace.ShapeCore(Trace.translate(Vec3f0(-1, 0.31, -98)), false),
    #     0.3f0, 360f0,
    # ), plastic_purple)
    triangle_meshes, triangles = Trace.load_triangle_mesh(
        Trace.ShapeCore(Trace.translate(Vec3f0(5, -1.49, -100)), false),
        model,
    )
    floor_triangles = Trace.create_triangle_mesh(
        Trace.ShapeCore(Trace.translate(Vec3f0(-10, 0, -87)), false),
        2, UInt32[1, 2, 3, 1, 4, 3],
        4,
        [
            Point3f0(0, 0, 0), Point3f0(0, 0, -30),
            Point3f0(30, 0, -30), Point3f0(30, 0, 0),
        ],
        [
            Trace.Normal3f0(0, 1, 0), Trace.Normal3f0(0, 1, 0),
            Trace.Normal3f0(0, 1, 0), Trace.Normal3f0(0, 1, 0),
        ],
    )

    primitives = Vector{Trace.GeometricPrimitive}(undef, 0)
    # push!(primitives, sphere_primitive1)
    # push!(primitives, sphere_primitive2)
    # push!(primitives, sphere_primitive3)
    for t in triangles
        push!(primitives, Trace.GeometricPrimitive(t, glass))
    end
    for t in floor_triangles
        push!(primitives, Trace.GeometricPrimitive(t, plastic))
    end

    bvh = Trace.BVHAccel(primitives, 1)
    # println("BVH World bounds $(Trace.world_bound(bvh))")

    from, to = Point3f0(0, 2, 0), Point3f0(-5, 0, 5)
    cone_angle, cone_δ_angle = 30f0, 10f0
    dir = Vec3f0(to - from) |> normalize
    dir, du, dv = Trace.coordinate_system(dir, Vec3f0(0f0))

    dir_to_z = Trace.Transformation(Mat4f0(
        du[1], du[2], du[3], 0f0,
        dv[1], dv[2], dv[3], 0f0,
        dir[1], dir[2], dir[3], 0f0,
        0f0, 0f0, 0f0, 1f0,
    ) |> transpose)
    light_to_world = (
        Trace.translate(Vec3f0(4.5, 0, -101))
        * Trace.translate(Vec3f0(from))
        * inv(dir_to_z)
    )

    lights = [
        Trace.SpotLight(
            light_to_world, Trace.RGBSpectrum(60f0),
            cone_angle, cone_angle - cone_δ_angle,
        ),
    ]

    scene = Trace.Scene(lights, bvh)

    resolution = Point2f0(1024)
    n_samples = 8
    ray_depth = 8

    look_point = Point3f0(-3, 0, -91)
    screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    filter = Trace.LanczosSincFilter(Point2f0(1f0), 3f0)

    ir = resolution .|> Int64
    film = Trace.Film(
        resolution, Trace.Bounds2(Point2f0(0), Point2f0(1)),
        filter, 1f0, 1f0, "./scenes/caustic-sppm-$(ir[1])x$(ir[2]).png",
    )
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f0(0, 150, 150), look_point, Vec3f0(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
    )

    # sampler = Trace.UniformSampler(n_samples)
    # integrator = Trace.WhittedIntegrator(camera, sampler, ray_depth)
    integrator = Trace.SPPMIntegrator(
        camera, 0.075f0, ray_depth, 10_000, 1_000_000,
    )
    scene |> integrator
end

render()
