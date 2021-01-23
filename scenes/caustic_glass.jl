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
        Trace.ConstantTexture(1.5f0),
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
    for t in floor_triangles[1:4]
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
        Trace.translate(Vec3f0(-5, 10, -90)),
        Trace.RGBSpectrum(intensity),
    )]
    scene = Trace.Scene(lights, bvh)

    resolution = Point2f0(1024)
    n_samples = 8
    ray_depth = 8

    # look_point = Trace.bounding_sphere(Trace.world_bound(bvh))[1]
    # look_point *= Point3f0(0, 0, 1)
    # look_point -= Point3f0(0.5, 2, 0)
    look_point = Point3f0(-0.5, -2.0, -101.750374)
    println("Look point $look_point")

    screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    filter = Trace.LanczosSincFilter(Point2f0(1f0), 3f0)

    screen_separation = _create_screen_separation(Threads.nthreads())
    println("Screen separation:")
    display(screen_separation)
    println()

    println("[!] Utilizing $(Threads.nthreads()) threads")

    t1 = time()
    bar = Progress(length(screen_separation), 1)
    Threads.@threads for (ss, tmp_path) in screen_separation
        film = Trace.Film(resolution, ss, filter, 1f0, 1f0, tmp_path)
        camera = Trace.PerspectiveCamera(
            Trace.look_at(Point3f0(0, 60, 100), look_point, Vec3f0(0, 1, 0)),
            screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
        )

        sampler = Trace.UniformSampler(n_samples)
        integrator = Trace.WhittedIntegrator(camera, sampler, ray_depth)
        scene |> integrator

        next!(bar)
    end
    t2 = time()
    println("Total execution time $(t2 - t1)")

    final_image = _assemble(resolution, screen_separation)
    ir = resolution .|> Int64
    save("scenes/caustic-glass2-$(ir[1])x$(ir[2]).png", final_image)
end

function _create_screen_separation(n_parts)
    if n_parts == 1
        return [(Trace.Bounds2(Point2f0(0), Point2f0(1)), "scenes/tmp.png")]
    end
    if n_parts % 4 != 0
        error("Number of parts should be divisible by 4, but is instead $n_parts.")
    end

    step = 1 / (n_parts / 2)
    steps = length(0:step:1) - 2
    screen_separation = []
    k = 1
    for i in 0:steps, j in 0:steps
        push!(screen_separation, (
            Trace.Bounds2(
                Point2f0(i * step, j * step),
                Point2f0((i + 1) * step, (j + 1) * step),
            ),
            "scenes/tmp-$k.png",
        ))
        k += 1
    end
    screen_separation
end

function _assemble(resolution::Point2f0, screen_separation::Vector)
    image = Array{RGB, 2}(undef, Int64(resolution[1]), Int64(resolution[2]))
    for (b, f) in screen_separation
        p_min = Point2f0(1 - b.p_max[2], b.p_min[1])
        p_max = Point2f0(1 - b.p_min[2], b.p_max[1])
        sub_range_min = Int64.(p_min * resolution)
        sub_range_max = Int64.(p_max * resolution)

        sub_image = f |> load
        image[
            sub_range_min[1] + 1:sub_range_max[1],
            sub_range_min[2] + 1:sub_range_max[2],
        ] = sub_image
    end
    image
end

render()
