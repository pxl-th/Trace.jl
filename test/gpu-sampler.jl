using GeometryBasics, LinearAlgebra, Trace, BenchmarkTools
using ImageShow
using Makie
using KernelAbstractions
import KernelAbstractions as KA
using KernelAbstractions.Extras.LoopInfo: @unroll

using AMDGPU
ArrayType = ROCArray
# using CUDA
# ArrayType = CuArray

include("./../src/gpu-support.jl")

LowSphere(radius, contact=Point3f(0)) = Sphere(contact .+ Point3f(0, 0, radius), radius)

function tmesh(prim, material)

    prim =  prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Trace.create_triangle_mesh(mesh)
    return Trace.GeometricPrimitive(m, material)
end

material_red = Trace.MatteMaterial(
    Trace.ConstantTexture(Trace.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
    Trace.ConstantTexture(0.0f0),
)


begin
    s1 = tmesh(LowSphere(0.5f0), material_red)
    s2 = tmesh(LowSphere(0.3f0, Point3f(0.5, 0.5, 0)), material_red)
    s3 = tmesh(LowSphere(0.3f0, Point3f(-0.5, 0.5, 0)), material_red)
    s4 = tmesh(LowSphere(0.4f0, Point3f(0, 1.0, 0)), material_red)

    ground = tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, 0.01)), material_red)
    back = tmesh(Rect3f(Vec3f(-5, -3, 0), Vec3f(10, 0.01, 10)), material_red)
    l = tmesh(Rect3f(Vec3f(-2, -5, 0), Vec3f(0.01, 10, 10)), material_red)
    r = tmesh(Rect3f(Vec3f(2, -5, 0), Vec3f(0.01, 10, 10)), material_red)
    bvh = Trace.BVHAccel([s1, s2, s3, s4, ground, back, l, r]);
    res = 512
    resolution = Point2f(res)
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
    lights = (
        # Trace.PointLight(Vec3f(0, -1, 2), Trace.RGBSpectrum(22.0f0)),
        Trace.PointLight(Vec3f(0, 0, 2), Trace.RGBSpectrum(10.0f0)),
        Trace.PointLight(Vec3f(0, 3, 3), Trace.RGBSpectrum(25.0f0)),
    )
    scene = Trace.Scene(lights, bvh)
    img = zeros(RGBf, res, res)
end

# ray = Trace.Ray(o=Point3f(0.5, 0.5, 1.0), d=Vec3f(0.0, 0.0, -1.0))
# l = Trace.RGBSpectrum(0.0f0)
# open("test3.llvm", "w") do io
#     code_llvm(io, simple_shading, typeof.((bvh, bvh.primitives[1], Trace.RayDifferentials(ray), Trace.SurfaceInteraction(), l, 1, 1, lights)))
# end

@inline function trace_pixel(camera, scene, xy)
    pixel = Point2f(Tuple(xy))
    s = Trace.UniformSampler(8)
    camera_sample = @inline Trace.get_camera_sample(s, pixel)
    ray, ω = Trace.generate_ray_differential(camera, camera_sample)
    if ω > 0.0f0
        l = @inline Trace.li(s, 5, ray, scene, 1)
    end
    return l
end

@inline function trace_pixel(camera, scene, xy)
    pixel = Point2f(reverse(Tuple(xy)))
    s = Trace.UniformSampler(8)
    camera_sample = @inline Trace.get_camera_sample(s, pixel)
    ray, ω = Trace.generate_ray_differential(camera, camera_sample)
    l = Trace.RGBSpectrum(0.0f0)
    if ω > 0.0f0
        hit, prim, si = Trace.intersect!(scene, ray)
        if hit
            m = Trace.get_material(scene, prim)
            bsdf = m(si, false, Trace.Radiance)
            l = Trace.specular_reflect(
                bsdf, s, 8, ray,
                si, scene, 8,
            )
        end
    end
    return l
end

@kernel function ka_trace_image!(img, camera, scene)
    xy = @index(Global, Cartesian)
    if checkbounds(Bool, img, xy)
        l = trace_pixel(camera, scene, xy)
        @inbounds img[xy] = RGBf(l.c...)
    end
    nothing
end

function launch_trace_image!(img, camera, scene)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    kernel!(img, camera, scene, ndrange=size(img), workgroupsize=(16, 16))
    KA.synchronize(backend)
    return img
end

preserve = []
gpu_scene = to_gpu(ArrayType, scene; preserve=preserve);
gpu_img = ArrayType(zeros(RGBf, res, res));
# launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
launch_trace_image!(gpu_img, cam, gpu_scene);
Array(gpu_img)

launch_trace_image!(img, cam, scene)
GLMakie.activate!(float=true)
image(img)

ray = Trace.RayDifferentials(Trace.Ray(o=Point3f(0.5, 0.5, 1.0), d=Vec3f(0.0, 0.0, -1.0)))
open("le-wt.jl", "w") do io
    code_warntype(io, Trace.li, typeof.((Trace.UniformSampler(8), 5, ray, scene, 1)))
end


function launch_trace_image_ir!(img, camera, scene)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    open("test.ir", "w") do io
        try
            @device_code_llvm io begin
                kernel!(img, camera, scene, ndrange=size(img), workgroupsize=(16, 16))
            end
        catch e
            println(e)
        end
    end
    KA.synchronize(backend)
    return img
end
launch_trace_image_ir!(gpu_img, cam, gpu_scene);

code_llvm(Trace.intersect!, (typeof(bvh), Trace.RayDifferentials))


function trace_image!(img, camera, scene)
    for xy in CartesianIndices(size(img))
        @inbounds img[xy] = RGBf(trace_pixel(camera, scene, xy).c...)
    end
    return img
end


@time launch_trace_image!(img, cam, scene)
