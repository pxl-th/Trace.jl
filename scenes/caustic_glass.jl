using GeometryBasics
using Trace

function render()
    model = raw"C:\Users\tonys\projects\pbrt-v3-scenes\caustic-glass\geometry\mesh_00001.ply"

    red = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 0f0, 0f0)),
        Trace.ConstantTexture(0.5f0),
    )
    white = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0, 1f0, 1f0)),
        Trace.ConstantTexture(0f0),
    )
    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(0f0),
        Trace.ConstantTexture(0f0),
        Trace.ConstantTexture(1.5f0),
        true,
    )

    transformation = Trace.translate(Vec3f0(5, -1.49, -100))
    core = Trace.ShapeCore(transformation, false)
    triangle_meshes, triangles = Trace.load_triangle_mesh(core, model)
    triangle_primitives = [Trace.GeometricPrimitive(t, glass) for t in triangles]

    floor_triangles = Trace.create_triangle_mesh(
        Trace.ShapeCore(Trace.translate(Vec3f0(-2, 0, -86)), false),
        4,
        UInt32[
            1, 2, 3,
            1, 4, 3,
            2, 3, 5,
            6, 5, 3,
        ],
        6,
        [
            Point3f0(0, 0, 0), Point3f0(0, 0, -15),
            Point3f0(6, 0, -15), Point3f0(6, 0, 0),
            Point3f0(0, 6, -15), Point3f0(6, 6, -15),
        ],
        [
            Trace.Normal3f0(0, 1, 0), Trace.Normal3f0(0, 1, 0),
            Trace.Normal3f0(0, 1, 0), Trace.Normal3f0(0, 1, 0),
            Trace.Normal3f0(0, 0, 1), Trace.Normal3f0(0, 0, 1),
        ],
    )
    for t in floor_triangles
        push!(triangle_primitives, Trace.GeometricPrimitive(t, white))
    end

    bvh = Trace.BVHAccel(triangle_primitives, 1)
    println("BVH World bounds $(Trace.world_bound(bvh))")

    lights = [Trace.PointLight(
        Trace.translate(Vec3f0(-5, 10, -70)),
        Trace.RGBSpectrum(Float32(680 * π)),
    )]
    scene = Trace.Scene(lights, bvh)

    resolution = Point2f0(1024)
    filter = Trace.LanczosSincFilter(Point2f0(1f0), 3f0)
    film = Trace.Film(
        resolution, Trace.Bounds2(Point2f0(0f0), Point2f0(1f0)),
        filter, 1f0, 1f0,
        "scenes/caustic-glass-$(Int64(resolution[1]))x$(Int64(resolution[2])).png",
    )
    screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    # look_point = Trace.bounding_sphere(Trace.world_bound(bvh))[1]
    # look_point *= Point3f0(0, 0, 1)
    # look_point -= Point3f0(0.5, 2, 0)
    look_point = Point3f0(-0.5, -2.0, -101.750374)
    println("Look point $look_point")
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f0(0, 50, 100), look_point, Vec3f0(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
    )

    sampler = Trace.UniformSampler(8)
    integrator = Trace.WhittedIntegrator(camera, sampler, 8)
    scene |> integrator
end

render()
