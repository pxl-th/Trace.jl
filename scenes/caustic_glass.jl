using GeometryBasics
using Trace
using ProgressMeter
using Printf
using FileIO
using ImageCore
using LinearAlgebra

function render()
    model = raw"./scenes/models/caustic-glass.ply"

    red = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 0f0, 0f0)),
        Trace.ConstantTexture(0f0),
    )
    white = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 1f0, 1f0)),
        Trace.ConstantTexture(0f0),
    )
    mirror = Trace.MirrorMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 1f0, 1f0)),
    )
    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(0f0),
        Trace.ConstantTexture(0f0),
        Trace.ConstantTexture(1.25f0),
        true,
    )
    plastic = Trace.PlasticMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
        Trace.ConstantTexture(0.010408001f0),
        true,
    )
    plastic_red = Trace.PlasticMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.1f0, 0.01f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
        Trace.ConstantTexture(0.010408001f0),
        true,
    )

    sphere_primitive1 = Trace.GeometricPrimitive(Trace.Sphere(
        Trace.ShapeCore(Trace.translate(Vec3f0(1.3, 2.2, -98.2)), false),
        0.2f0, 360f0,
    ), plastic_red)
    sphere_primitive2 = Trace.GeometricPrimitive(Trace.Sphere(
        Trace.ShapeCore(Trace.translate(Vec3f0(2, 0.26, -96)), false),
        0.25f0, 360f0,
    ), plastic_red)

    # triangle_meshes, triangles = Trace.load_triangle_mesh(
    #     Trace.ShapeCore(Trace.translate(Vec3f0(5, -1.49, -100)), false),
    #     model,
    # )

    floor_triangles = Trace.create_triangle_mesh(
        Trace.ShapeCore(Trace.translate(Vec3f0(-2, 0, -87)), false),
        2,
        UInt32[
            1, 2, 3,
            1, 4, 3,
        ],
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

    # n_primitives = length(triangles) + length(floor_triangles) + 2
    n_primitives = length(floor_triangles) + 2
    primitives = Vector{Trace.GeometricPrimitive}(undef, n_primitives)
    filled = 1

    primitives[filled] = sphere_primitive1
    filled += 1
    primitives[filled] = sphere_primitive2
    filled += 1
    # for t in triangles
    #     primitives[filled] = Trace.GeometricPrimitive(t, glass)
    #     filled += 1
    # end
    for t in floor_triangles
        primitives[filled] = Trace.GeometricPrimitive(t, plastic)
        filled += 1
    end
    # for t in floor_triangles[3:4]
    #     primitives[filled] = Trace.GeometricPrimitive(t, glass)
    #     filled += 1
    # end
    @assert filled == n_primitives + 1

    bvh = Trace.BVHAccel(primitives, 1)
    println("BVH World bounds $(Trace.world_bound(bvh))")

    intensity = 150f0 * π

    from, to = Point3f0(1, 7, -1), Point3f0(-5, 0, 5)
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
        Trace.translate(Vec3f0(6, 0, -102))
        * Trace.translate(Vec3f0(from))
        * inv(dir_to_z)
    )

    lights = [
        # Trace.PointLight(
        #     Trace.translate(Vec3f0(8, 7, -107)), Trace.RGBSpectrum(intensity),
        # ),
        Trace.SpotLight(
            light_to_world, Trace.RGBSpectrum(intensity),
            cone_angle, cone_angle - cone_δ_angle,
        ),
        # Trace.PointLight(
        #     Trace.translate(Vec3f0(0, 10, -95)),
        #     Trace.RGBSpectrum(intensity / 4f0),
        # ),
    ]

    scene = Trace.Scene(lights, bvh)

    resolution = Point2f0(512)
    n_samples = 8
    ray_depth = 8

    # look_point = Trace.bounding_sphere(Trace.world_bound(bvh))[1]
    # look_point *= Point3f0(0, 0, 1)
    # look_point -= Point3f0(0.5, 2, 0)
    # look_point = Point3f0(-0.5, -2.0, -101.750374)
    look_point = Point3f0(-1.5, -2.0, -100.5)
    println("Look point $look_point")

    screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    filter = Trace.LanczosSincFilter(Point2f0(1f0), 3f0)

    ir = resolution .|> Int64
    film = Trace.Film(
        resolution, Trace.Bounds2(Point2f0(0), Point2f0(1)),
        filter, 1f0, 1f0, "./scenes/caustic-debug-sppm-$(ir[1])x$(ir[2]).png",
    )
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f0(0, 60, 100), look_point, Vec3f0(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
    )

    # sampler = Trace.UniformSampler(n_samples)
    # integrator = Trace.WhittedIntegrator(camera, sampler, ray_depth)
    integrator = Trace.SPPMIntegrator(camera, 0.075f0, ray_depth, 1, 100_000, 1)
    scene |> integrator
end

render()
