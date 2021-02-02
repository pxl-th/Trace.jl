using GeometryBasics
using Trace
using ProgressMeter
using Printf
using FileIO
using ImageCore

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

    sphere_primitive1 = Trace.GeometricPrimitive(Trace.Sphere(
        Trace.ShapeCore(Trace.translate(Vec3f0(1.3, 2.2, -98.2)), false),
        0.2f0, 360f0,
    ), red)
    sphere_primitive2 = Trace.GeometricPrimitive(Trace.Sphere(
        Trace.ShapeCore(Trace.translate(Vec3f0(2, 0.26, -96)), false),
        0.25f0, 360f0,
    ), red)

    transformation = Trace.translate(Vec3f0(5, -1.49, -100))
    core = Trace.ShapeCore(transformation, false)
    triangle_meshes, triangles = Trace.load_triangle_mesh(core, model)

    floor_triangles = Trace.create_triangle_mesh(
        Trace.ShapeCore(Trace.translate(Vec3f0(-2, 0, -87)), false),
        2,
        UInt32[
            1, 2, 3,
            1, 4, 3,
            # 2, 3, 5,
            # 6, 5, 3,
        ],
        4,
        [
            Point3f0(0, 0, 0), Point3f0(0, 0, -30),
            Point3f0(30, 0, -30), Point3f0(30, 0, 0),
            # Point3f0(0, 6, -15), Point3f0(6, 6, -15),
        ],
        [
            Trace.Normal3f0(0, 1, 0), Trace.Normal3f0(0, 1, 0),
            Trace.Normal3f0(0, 1, 0), Trace.Normal3f0(0, 1, 0),
            # Trace.Normal3f0(0, 0, 1), Trace.Normal3f0(0, 0, 1),
        ],
    )

    n_primitives = length(triangles) + length(floor_triangles) + 2
    primitives = Vector{Trace.GeometricPrimitive}(undef, n_primitives)
    filled = 1

    primitives[filled] = sphere_primitive1
    filled += 1
    primitives[filled] = sphere_primitive2
    filled += 1
    for t in triangles
        primitives[filled] = Trace.GeometricPrimitive(t, glass)
        filled += 1
    end
    for t in floor_triangles
        primitives[filled] = Trace.GeometricPrimitive(t, white)
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
    lights = [Trace.PointLight(
        # Trace.translate(Vec3f0(-5, 10, -90)),
        Trace.translate(Vec3f0(8, 10, -108)),
        Trace.RGBSpectrum(intensity),
    )]
    scene = Trace.Scene(lights, bvh)

    resolution = Point2f0(128)
    # n_samples = 8
    # ray_depth = 8

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
        filter, 1f0, 1f0, "./scenes/caustic-sppm-$(ir[1])x$(ir[2]).png",
    )
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f0(0, 60, 100), look_point, Vec3f0(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
    )

    # sampler = Trace.UniformSampler(n_samples)
    # integrator = Trace.WhittedIntegrator(camera, sampler, ray_depth)
    # integrator = Trace.SPPMIntegrator(camera, 0.075f0, 8, 100, 10_000, 1)
    integrator = Trace.SPPMIntegrator(camera, 0.025f0, 8, 100, -1, 1)
    scene |> integrator
end

render()
