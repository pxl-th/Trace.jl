using GeometryBasics
using Trace
using FileIO
using ImageCore
using BenchmarkTools
using Makie, FileIO, ImageShow

function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    triangles = Trace.create_triangle_mesh(mesh, Trace.ShapeCore())
    return [Trace.GeometricPrimitive(t, material) for t in triangles]
end


function random_spheres()
    triangles = []
    append!(triangles, tmesh(Sphere(Point3(0, -1000, 0), 1000.0), Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(0.5f0)))))

    function rand_material()
        p = rand()
        if p < 0.8
            Trace.MatteMaterial(
                Trace.ConstantTexture(Trace.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
                Trace.ConstantTexture(0.0f0),
            )
        elseif p < 0.95
            rf = rand(Float32)
            Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(rf)))
        else
            Trace.PlasticMaterial(
                Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
                Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
                Trace.ConstantTexture(0.010408001f0),
                true,
            )
        end
    end

    for a in -11:10, b in -11:10
        center = Point3f(a + 0.9rand(), 0.2, b + 0.9rand())
        if norm(center - Point3f(4, 0.2, 0)) > 0.9
            append!(triangles, tmesh(Sphere(center, 0.2), rand_material()))
        end
    end
    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(1.5f0),
        true,
    )

    append!(triangles, tmesh(Sphere(Point3f(0, 1, 0), 1.0), glass))
    append!(triangles, tmesh(Sphere(Point3f(-4, 1, 0), 1.0), rand_material()))
    append!(triangles, tmesh(Sphere(Point3f(4, 1, 0), 1.0), rand_material()))

    return map(identity, triangles)
end


begin
    bvh = Trace.BVHAccel(random_spheres(), 1);

    lights = [
        # Trace.PointLight(Vec3f(0, -1, 2), Trace.RGBSpectrum(22.0f0)),
        Trace.PointLight(Vec3f(0, 0, 2), Trace.RGBSpectrum(10.0f0)),
        Trace.PointLight(Vec3f(0, 3, 3), Trace.RGBSpectrum(15.0f0)),
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
    integrator = Trace.WhittedIntegrator(cam, Trace.UniformSampler(8), 10)
    @time integrator(scene)
    img = reverse(film.framebuffer, dims=1)
end
