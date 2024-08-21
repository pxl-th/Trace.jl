using GeometryBasics
using Trace
using FileIO
using ImageCore
using BenchmarkTools
using Makie, FileIO, ImageShow


catmesh = load(Makie.assetpath("cat.obj"))
img = load(Makie.assetpath("diffusemap.png"))
m = normal_mesh(Tesselation(Sphere(Point3f(0), 1), 32))

function tmesh(prim, material)
    prim =  prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    triangles = Trace.create_triangle_mesh(mesh, Trace.ShapeCore())
    return [Trace.GeometricPrimitive(t, material) for t in triangles]
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
    s4 = tmesh(LowSphere(0.4f0, Point3f(0.0, 1.0, 0)), material_white)
    ground = tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, -0.1)), mirror)
    left = tmesh(Rect3f(Vec3f(-3, 0, 0), Vec3f(-0.1, 10, -0.1)), mirror)

    bvh = Trace.BVHAccel([s1..., s2..., s3..., s4..., ground...], 1);

    lights = [
        Trace.PointLight(Vec3f(0, 2.5, 4), Trace.RGBSpectrum(60.0f0)),
        Trace.PointLight(Vec3f(-5, 2, 0), Trace.RGBSpectrum(60.0f0)),
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
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f(0, 3, 3), Point3f(0, 0, 0), Vec3f(0, 0, 1)),
        screen_window, 0.0f0, 1.0f0, 0.0f0, 1.0f6, 45.0f0, film,
    )
    integrator = Trace.WhittedIntegrator(camera, Trace.UniformSampler(8), 1)
    integrator(scene)
    img = reverse(film.framebuffer, dims=1)
end
x = Trace.scale(2, 2, 2)
x.inv_m == inv(x.m)
GLMakie.activate!(inline=true)
# Computer projective camera transformations.
resolution = Point2f(1024)
resolution = Trace.scale(resolution[1], resolution[2], 1)

window_width = screen_window.p_max .- screen_window.p_min
inv_bounds = Trace.scale((1.0f0 ./ window_width)..., 1)

offset = Trace.translate(Vec3f(
    -screen_window.p_min..., 0.0f0,
))

ray = camera.core.raster_to_camera(Point3f(1024 / 2, 1024 / 2, 0))
camera.core.screen_to_raster(camera.core.camera_to_screen(ray))

camera.core.screen_to_raster(Point3f(-1, -1, 0))
camera.core.screen_to_raster(Point3f(1, 1, 0))

camera.core.raster_to_screen(Point3f(1024, 1024, 0))

camera.core.screen_to_raster(Point3f(-1, -1, 0))# == (0, 0, 0)
camera.core.screen_to_raster(Point3f(1, 1, 0))# == (1024, 1024, 0)

camera.core.raster_to_screen(Point3f(0, 0, 0)) # == (0, 0, 0)
camera.core.screen_to_raster(Point3f(1, 1, 0)) # == (1024, 1024, 0)


res_t = Trace.scale(resolution..., 1)
window_width = screen_window.p_max .- screen_window.p_min
inv_bounds = Trace.scale((1f0 ./ window_width)..., 1)

offset = Trace.translate(Vec3f(
    (-screen_window.p_min)..., 0f0,
))

inv(res_t.m)
screen_to_raster = res_t * inv_bounds * offset

screen_to_raster(Point3f(1, 1, 0))

raster_to_screen = inv(offset) *  inv(inv_bounds) * inv(res_t)

raster_to_screen(Point3f(1024, 1024, 0))
