using GeometryBasics
using LinearAlgebra
using Trace

function render()
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

    model = "./scenes/models/caustic-glass.ply"
    triangle_meshes, triangles = Trace.load_triangle_mesh(
        model, Trace.ShapeCore(Trace.translate(Vec3f0(5, -1.49, -100)), false),
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
    for t in triangles
        push!(primitives, Trace.GeometricPrimitive(t, glass))
    end
    for t in floor_triangles
        push!(primitives, Trace.GeometricPrimitive(t, plastic))
    end

    bvh = Trace.BVHAccel(primitives, 1)

    from, to = Point3f0(0, 2, 0), Point3f0(-5, 0, 5)
    cone_angle, cone_δ_angle = 30f0, 10f0
    dir = Vec3f0(to - from) |> normalize
    dir, du, dv = Trace.coordinate_system(dir)

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

    resolution = Point2f0(256)
    n_samples = 8
    ray_depth = 5

    look_point = Point3f0(-3, 0, -91)
    screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    filter = Trace.LanczosSincFilter(Point2f0(1f0), 3f0)

    ir = resolution .|> Int64
    film = Trace.Film(
        resolution, Trace.Bounds2(Point2f0(0), Point2f0(1)),
        filter, 1f0, 1f0,
        "./scenes/caustics-sppm-$(ir[1])x$(ir[2]).png",
    )
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f0(0, 150, 150), look_point, Vec3f0(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
    )

    integrator = Trace.SPPMIntegrator(camera, 0.075f0, ray_depth, 100, -1)
    scene |> integrator
end

render()
