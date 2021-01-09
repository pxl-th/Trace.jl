using GeometryBasics
using Trace

function simple()
    material = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 0f0, 0f0)),
        Trace.ConstantTexture(0f0),
    )
    material_tr = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 0f0, 0f0)),
        Trace.ConstantTexture(0f0),
    )
    material2 = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.2f0, 0.1f0, 1f0)),
        Trace.ConstantTexture(0f0),
    )
    mirror = Trace.MirrorMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 1f0, 1f0)),
    )

    core = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.017f0, 0.017f0, -3f0)), false,
    )
    sphere = Trace.Sphere(core, 0.006f0, 360f0)
    primitive = Trace.GeometricPrimitive(sphere, material)

    core2 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.007f0, 0.007f0, -3f0)), false,
    )
    sphere2 = Trace.Sphere(core2, 0.006f0, 360f0)
    primitive2 = Trace.GeometricPrimitive(sphere2, mirror)

    core3 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.004f0, 0.02f0, -3f0)), false,
    )
    sphere3 = Trace.Sphere(core3, 0.006f0, 360f0)
    primitive3 = Trace.GeometricPrimitive(sphere3, material2)

    core4 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.004f0, 0.023f0, -2.95f0)), false,
    )
    sphere4 = Trace.Sphere(core4, 0.004f0, 360f0)
    primitive4 = Trace.GeometricPrimitive(sphere4, material)

    triangles = Trace.create_triangle_mesh(
        Trace.ShapeCore(Trace.translate(Vec3f0(0, 0, -3.5)), false),
        1, UInt32[1, 2, 3],
        3, [Point3f0(0, 0, 0), Point3f0(0.02, 0, 0), Point3f0(0.02, 0.02, 0)],
        [Trace.Normal3f0(0, 0, 1), Trace.Normal3f0(0, 0, 1), Trace.Normal3f0(0, 0, 1)],
    )
    triangle_primitive = Trace.GeometricPrimitive(triangles[1], material_tr)

    bvh = Trace.BVHAccel([
        primitive, primitive2, primitive3, primitive4, triangle_primitive,
    ], 1,)
    for n in bvh.nodes
        display(n); println()
    end

    lights = [Trace.PointLight(
        Trace.translate(Vec3f0(0f0, -1f0, -1f0)),
        Trace.RGBSpectrum(Float32(4 * π)),
    )]
    scene = Trace.Scene(lights, bvh)

    resolution = Point2f0(512, 512)
    filter = Trace.LanczosSincFilter(Point2f0(1f0), 3f0)
    film = Trace.Film(
        resolution, Trace.Bounds2(Point2f0(0f0), Point2f0(1f0)),
        filter, 1f0, 1f0, "output.png",
    )
    screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    camera = Trace.PerspectiveCamera(
        Trace.Transformation(), screen, 0f0, 1f0, 0f0, 10f0, 45f0, film,
    )

    sampler = Trace.UniformSampler(8)
    integrator = Trace.WhittedIntegrator(camera, sampler, 2)
    scene |> integrator
end

simple()
