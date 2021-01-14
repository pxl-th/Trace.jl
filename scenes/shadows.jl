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

    core2 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.01, 0.002, -2.97)), false,
    )
    sphere2 = Trace.Sphere(core2, 0.01f0, 360f0)
    primitive2 = Trace.GeometricPrimitive(sphere2, mirror)

    core3 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.04, 0.002, -2.86)), false,
    )
    sphere3 = Trace.Sphere(core3, 0.01f0, 360f0)
    primitive3 = Trace.GeometricPrimitive(sphere3, material_blue)

    # TODO bvh construction failes when z coordinates are equal?
    core4 = Trace.ShapeCore(
        Trace.translate(Vec3f0(0.035, 0.002, -2.93)), false,
    )
    sphere4 = Trace.Sphere(core4, 0.01f0, 360f0)
    primitive4 = Trace.GeometricPrimitive(sphere4, material_red)

    triangles = Trace.create_triangle_mesh(
        Trace.ShapeCore(Trace.translate(Vec3f0(0, 0.021, -2.7)), false),
        2, UInt32[1, 2, 3, 1, 4, 3],
        4, [Point3f0(0, 0, 0), Point3f0(0, -0.001, -1), Point3f0(1, -0.001, -1), Point3f0(1, 0, 0)],
        [Trace.Normal3f0(0, -1, 0), Trace.Normal3f0(0, -1, 0), Trace.Normal3f0(0, -1, 0),
         Trace.Normal3f0(0, -1, 0), Trace.Normal3f0(0, -1, 0), Trace.Normal3f0(0, -1, 0)],
    )
    triangle_primitive = Trace.GeometricPrimitive(triangles[1], material_white)
    triangle_primitive2 = Trace.GeometricPrimitive(triangles[2], material_white)

    bvh = Trace.BVHAccel([
        primitive2,
        primitive3,
        primitive4,
        triangle_primitive,
        triangle_primitive2,
    ], 1)
    for n in bvh.nodes
        display(n); println()
    end

    lights = [Trace.PointLight(
        Trace.translate(Vec3f0(-1, -1, 0)),
        Trace.RGBSpectrum(Float32(10 * π)),
    )]
    scene = Trace.Scene(lights, bvh)

    resolution = Point2f0(64)
    filter = Trace.LanczosSincFilter(Point2f0(1f0), 3f0)
    film = Trace.Film(
        resolution, Trace.Bounds2(Point2f0(0f0), Point2f0(1f0)),
        filter, 1f0, 1f0,
        "scenes/shadows-$(Int64(resolution[1]))x$(Int64(resolution[2])).png",
    )
    screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f0(0, -1, 0), Point3f0(0, 0, -3), Vec3f0(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 10f6, 90f0, film,
    )

    sampler = Trace.UniformSampler(8)
    integrator = Trace.WhittedIntegrator(camera, sampler, 8)
    scene |> integrator
end

render()
