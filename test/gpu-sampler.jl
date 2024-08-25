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

begin

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
    plastic = Trace.PlasticMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.7399999857f0, 0.6399999857f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(0.2000000015f0, 0.2000000015f0, 0.2000000015f0)),
        Trace.ConstantTexture(0.0010408001f0),
        true,
    )
    prims = [
        tmesh(LowSphere(0.5f0), mirror),
        tmesh(LowSphere(0.3f0, Point3f(0.5, 0.5, 0)), material_blue),
        tmesh(LowSphere(0.3f0, Point3f(-0.5, 0.5, 0)), glass),
        tmesh(LowSphere(0.2f0, Point3f(0, 1.4, 0)), plastic),
        tmesh(LowSphere(0.4f0, Point3f(0, 1.0, 0)), material_red),
        tmesh(LowSphere(0.1f0, Point3f(0.5, 1.0, 0)), glass),
        tmesh(LowSphere(0.1f0, Point3f(-0.5, 1.0, 0)), glass),

        tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, 0.01)), material_white),
        tmesh(Rect3f(Vec3f(-5, -3, 0), Vec3f(10, 0.01, 10)), material_white),
        tmesh(Rect3f(Vec3f(-2, -5, 0), Vec3f(0.01, 10, 10)), material_white),
        tmesh(Rect3f(Vec3f(2, -5, 0), Vec3f(0.01, 10, 10)), material_white),
    ]
    bvh = Trace.BVHAccel(prims, 1)

    lights = (
        # Trace.PointLight(Vec3f(0, -1, 2), Trace.RGBSpectrum(22.0f0)),
        Trace.PointLight(Vec3f(0, 0, 2), Trace.RGBSpectrum(10.0f0)),
        Trace.PointLight(Vec3f(0, 3, 3), Trace.RGBSpectrum(20.0f0)),
    )
    scene = Trace.Scene([lights...], bvh)
    res = 1024
    resolution = Point2f(res)
    f = Trace.LanczosSincFilter(Point2f(1.0f0), 3.0f0)
    film = Trace.Film(resolution,
        Trace.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
        "shadows_sppm_res.png",
    )
    screen_window = Trace.Bounds2(Point2f(-1), Point2f(1))
    cam = Trace.PerspectiveCamera(
        Trace.look_at(Point3f(1, 3.4, 1.8), Point3f(-0.5, -4, -1), Vec3f(0, 0, 1)),
        screen_window, 0.0f0, 1.0f0, 0.0f0, 1.0f6, 45.0f0, film,
    )
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
    l = Trace.RGBSpectrum(0.0f0)
    sampler = Trace.UniformSampler(3)
    if ω > 0.0f0
        l = Trace.li_iterative(sampler, Int32(8), ray, scene)
        # l = Trace.li(sampler, Int32(8), ray, scene, Int32(1))
    end
    return l
end

@kernel function ka_trace_image!(img, camera, scene)
    _idx = @index(Global)
    idx = _idx % Int32
    if checkbounds(Bool, img, idx)

        cols = size(img, 2) % Int32
        row = (idx - Int32(1)) ÷ cols + Int32(1)
        col = (idx - Int32(1)) % cols + Int32(1)
        l = trace_pixel(camera, scene, (row, cols - col))
        img[idx] = RGBf(l.c...)
    end
    nothing
end

function launch_trace_image!(img, camera, scene)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    kernel!(img, camera, scene, ndrange=size(img))
    KA.synchronize(backend)
    return img
end
preserve = []
gpu_scene = to_gpu(ArrayType, scene; preserve=preserve);
gpu_img = ArrayType(zeros(RGBf, res, res));
launch_trace_image!(gpu_img, cam, gpu_scene);
Array(gpu_img)
#95.787 ms (912 allocations: 27.22 KiB)
img = zeros(RGBf, res, res)
@time launch_trace_image!(img, cam, scene)
# 4.5s (CPU)
# 3s (GPU)

# 0.912289 seconds (2.12 M allocations: 178.238 MiB, 2.60% gc time)


function single_trace_image!(img, camera, scene)
    @inbounds for idx in eachindex(img)
        cols = size(img, 2) % Int32
        row = (idx - Int32(1)) ÷ cols + Int32(1)
        col = (idx - Int32(1)) % cols + Int32(1)
        l = trace_pixel(camera, scene, (row, cols - col))
        img[idx] = RGBf(l.c...)
    end
    return img
end

nothing
# 81.839 ms (233 allocations: 86.09 KiB)

# ray = Trace.RayDifferentials(Trace.Ray(o=Point3f(0.5, 0.5, 1.0), d=Vec3f(0.0, 0.0, -1.0)))
# open("le-wt.jl", "w") do io
#     code_warntype(io, Trace.li, typeof.((Trace.UniformSampler(8), 5, ray, scene, 1)))
# end


# function launch_trace_image_ir!(img, camera, scene)
#     backend = KA.get_backend(img)
#     kernel! = ka_trace_image!(backend)
#     open("test.ir", "w") do io
#         try
#             @device_code_llvm io begin
#                 kernel!(img, camera, scene, ndrange=size(img), workgroupsize=(16, 16))
#             end
#         catch e
#             println(e)
#         end
#     end
#     KA.synchronize(backend)
#     return img
# end
# launch_trace_image_ir!(gpu_img, cam, gpu_scene);

# code_llvm(Trace.intersect!, (typeof(bvh), Trace.RayDifferentials))


# function trace_image!(img, camera, scene)
#     for xy in CartesianIndices(size(img))
#         @inbounds img[xy] = RGBf(trace_pixel(camera, scene, xy).c...)
#     end
#     return img
# end


# @time launch_trace_image!(img, cam, scene)

s = Trace.UniformSampler(8)
pixel = Point2f(512, 512)
camera_sample = Trace.get_camera_sample(s, pixel)
ray, ω = Trace.generate_ray_differential(cam, camera_sample)

@btime Trace.li_iterative(s, 8, ray, scene)

MVector((ray, Int32(0), Trace.RGBSpectrum(0.0f0)), (ray, Int32(0), Trace.RGBSpectrum(0.0f0)))
