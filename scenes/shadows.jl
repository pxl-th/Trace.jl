using GeometryBasics
using Trace

function render()
    material_red = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 0f0, 0f0)),
        Trace.ConstantTexture(0f0),
    )
    material_blue = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.2f0, 0.1f0, 1f0)),
        Trace.ConstantTexture(0f0),
    )
    material_white = Trace.MatteMaterial(
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
        Trace.ConstantTexture(1.5f0),
        true,
    )

    core = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.3, 0.11, -2.2)), false,
    )
    sphere = Trace.Sphere(core, 0.1f0, 360f0)
    primitive = Trace.GeometricPrimitive(sphere, glass)

    core2 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.2, 0.11, -2.6)), false,
    )
    sphere2 = Trace.Sphere(core2, 0.1f0, 360f0)
    primitive2 = Trace.GeometricPrimitive(sphere2, material_blue)

    core3 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.7, 0.31, -2.8)), false,
    )
    sphere3 = Trace.Sphere(core3, 0.3f0, 360f0)
    primitive3 = Trace.GeometricPrimitive(sphere3, mirror)

    core4 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.7, 0.11, -2.3)), false,
    )
    sphere4 = Trace.Sphere(core4, 0.1f0, 360f0)
    primitive4 = Trace.GeometricPrimitive(sphere4, material_red)

    triangles = Trace.create_triangle_mesh(
        Trace.ShapeCore(Trace.translate(Vec3f0(0, 0, -2)), false),
        4,
        UInt32[
            1, 2, 3,
            1, 4, 3,
            2, 3, 5,
            6, 5, 3,
        ],
        6,
        [
            Point3f0(0, 0, 0), Point3f0(0, 0, -1),
            Point3f0(1, 0, -1), Point3f0(1, 0, 0),
            Point3f0(0, 1, -1), Point3f0(1, 1, -1),
        ],
        [
            Trace.Normal3f0(0, 1, 0), Trace.Normal3f0(0, 1, 0),
            Trace.Normal3f0(0, 1, 0), Trace.Normal3f0(0, 1, 0),
            Trace.Normal3f0(0, 0, 1), Trace.Normal3f0(0, 0, 1),
        ],
    )
    triangle_primitive = Trace.GeometricPrimitive(triangles[1], material_white)
    triangle_primitive2 = Trace.GeometricPrimitive(triangles[2], material_white)
    triangle_primitive3 = Trace.GeometricPrimitive(triangles[3], material_white)
    triangle_primitive4 = Trace.GeometricPrimitive(triangles[4], material_white)

    bvh = Trace.BVHAccel([
        primitive,
        primitive2,
        primitive3,
        primitive4,
        triangle_primitive,
        triangle_primitive2,
        triangle_primitive3,
        triangle_primitive4,
    ], 1)
    for n in bvh.nodes
        display(n); println()
    end

    lights = [Trace.PointLight(
        Trace.translate(Vec3f0(-1, 1, 0)),
        Trace.RGBSpectrum(Float32(10 * π)),
    )]
    scene = Trace.Scene(lights, bvh)

    resolution = Point2f0(64)
    filter = Trace.LanczosSincFilter(Point2f0(1f0), 3f0)
    film = Trace.Film(
        resolution, Trace.Bounds2(Point2f0(0f0), Point2f0(1f0)),
        filter, 1f0, 1f0,
        "scenes/shadows-sppm-$(Int64(resolution[1]))x$(Int64(resolution[2])).png",
    )
    screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f0(0, 15, 50), Point3f0(0, 0, -2), Vec3f0(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
    )

    # sampler = Trace.UniformSampler(8)
    # integrator = Trace.WhittedIntegrator(camera, sampler, 8)
    integrator = Trace.SPPMIntegrator(camera, 0.025f0, 8, 10)
    scene |> integrator
end

render()
