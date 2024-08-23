using GeometryBasics
using Trace
using FileIO
using ImageCore
using BenchmarkTools
using FileIO, ImageShow

function tmesh(prim, material)
    prim =  prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Trace.create_triangle_mesh(mesh, Trace.ShapeCore())
    return Trace.GeometricPrimitive(m, material)
end

material_red = Trace.MatteMaterial(
    Trace.ConstantTexture(Trace.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
    Trace.ConstantTexture(0.0f0),
)
material_blue = Trace.MatteMaterial(
    Trace.ConstantTexture(Trace.RGBSpectrum(0.251f0, 0.388f0, 0.847f0)),
    Trace.ConstantTexture(0.0f0),
)
material_white = Trace.MatteMaterial(
    Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
    Trace.ConstantTexture(0.0f0),
)
mirror = Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)))
glass = Trace.GlassMaterial(
    Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
    Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
    Trace.ConstantTexture(0.0f0),
    Trace.ConstantTexture(0.0f0),
    Trace.ConstantTexture(1.5f0),
    true,
)

LowSphere(radius, contact=Point3f(0)) = Sphere(contact .+ Point3f(0, 0, radius), radius)

begin
    s1 = tmesh(LowSphere(0.5f0), material_white)
    s2 = tmesh(LowSphere(0.3f0, Point3f(0.5, 0.5, 0)), material_blue)
    s3 = tmesh(LowSphere(0.3f0, Point3f(-0.5, 0.5, 0)), mirror)
    s4 = tmesh(LowSphere(0.4f0, Point3f(0, 1.0, 0)), glass)

    ground = tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, 0.01)), mirror)
    back = tmesh(Rect3f(Vec3f(-5, -3, 0), Vec3f(10, 0.01, 10)), material_white)
    l = tmesh(Rect3f(Vec3f(-2, -5, 0), Vec3f(0.01, 10, 10)), material_red)
    r = tmesh(Rect3f(Vec3f(2, -5, 0), Vec3f(0.01, 10, 10)), material_blue)

    bvh = Trace.BVHAccel([s1, s2, s3, s4, ground, back, l, r], 1);

    lights = [
        # Trace.PointLight(Vec3f(0, -1, 2), Trace.RGBSpectrum(22.0f0)),
        Trace.PointLight(Vec3f(0, 0, 2), Trace.RGBSpectrum(10.0f0)),
        Trace.PointLight(Vec3f(0, 3, 3), Trace.RGBSpectrum(25.0f0)),
    ]
    scene = Trace.Scene(lights, bvh);
    resolution = Point2f(1024)
    f = Trace.LanczosSincFilter(Point2f(1.0f0), 3.0f0)
    film = Trace.Film(resolution,
        Trace.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
        "shadows_sppm_res.png",
    )
    screen_window = Trace.Bounds2(Point2f(-1), Point2f(1))
    cam = Trace.PerspectiveCamera(
        Trace.look_at(Point3f(0, 4, 2), Point3f(0, -4, -1), Vec3f(0, 0, 1)),
        screen_window, 0.0f0, 1.0f0, 0.0f0, 1.0f6, 45.0f0, film,
    )
end

begin
    integrator = Trace.WhittedIntegrator(cam, Trace.UniformSampler(8), 5)
    @time integrator(scene, film)
    img = reverse(film.framebuffer, dims=1)
end
# 6.296157 seconds (17.64 k allocations: 19.796 MiB, 0.13% gc time, 45 lock conflicts)
# After more GPU optimizations
# 4.169616 seconds (17.37 k allocations: 19.777 MiB, 0.14% gc time, 20 lock conflicts)

camera_sample = Trace.get_camera_sample(integrator.sampler, Point2f(512))
ray, ω = Trace.generate_ray_differential(integrator.camera, camera_sample)

@btime Trace.intersect_p(bvh, ray)
@btime Trace.intersect!(bvh, ray)

###
# Int32 always
# 42.000 μs (1 allocation: 624 bytes)
# Tuple instead of vector for nodes_to_visit
# 43.400 μs (1 allocation: 624 bytes)
# AFTER GPU rework
# intersect!
# 40.500 μs (1 allocation: 368 bytes)
# intersect_p
# 11.500 μs (0 allocations: 0 bytes)

### LinearBVHLeaf as one type
# 5.247460 seconds (17.55 k allocations: 19.783 MiB, 46 lock conflicts)

# begin
#     integrator = Trace.SPPMIntegrator(cam, 0.075f0, 5, 1)
#     integrator(scene)
#     img = reverse(film.framebuffer, dims=1)
# end
