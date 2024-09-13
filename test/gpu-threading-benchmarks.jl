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
    img = zeros(RGBf, res, res)
end

@inline function get_camera_sample(p_raster::Point2)
    p_film = p_raster .+ rand(Point2f)
    p_lens = rand(Point2f)
    Trace.CameraSample(p_film, p_lens, rand(Float32))
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

@kernel function ka_trace_image!(img, camera, scene)
    xy = @index(Global, Cartesian)
    if checkbounds(Bool, img, xy)
        l = trace_pixel(camera, scene, xy)
        @_inbounds img[xy] = RGBf(l.c...)
    end
end

function launch_trace_image!(img, camera, scene)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    kernel!(img, camera, scene, lights, ndrange=size(img), workgroupsize=(16, 16))
    KA.synchronize(backend)
    return img
end
# using AMDGPU
# ArrayType = ROCArray
# using CUDA
# ArrayType = CuArray

# using Metal
# ArrayType = MtlArray

preserve = []
gpu_scene = to_gpu(ArrayType, scene; preserve=preserve);
gpu_img = ArrayType(zeros(RGBf, res, res));
# launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
launch_trace_image!(gpu_img, cam, gpu_scene);
launch_trace_image!(img, cam, scene, lights)
# 76.420 ms (234 allocations: 86.05 KiB)
# 75.973 ms (234 allocations: 86.05 KiB)
Array(gpu_img)

function cu_trace_image!(img, camera, bvh, lights)
    x = threadIdx().x
    y = threadIdx().y
    if checkbounds(Bool, img, (x, y))
        @_inbounds img[x, y] = trace_pixel(camera, bvh, (x,y), lights)
    end
end

k = some_kernel(img)
ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(k, size(img), (16, 16))
blocks = length(KA.blocks(iterspace))
threads = length(KA.workitems(iterspace))

function cu_launch_trace_image!(img, camera, bvh, lights)
    CUDA.@sync @cuda threads = length(img) cu_trace_image!(img, camera, bvh, lights)
    return img
end
cu_launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
Array(gpu_img)
# 380.081 ms (913 allocations: 23.55 KiB)
# CUDA (3070 mobile)
# 238.149 ms (46 allocations: 6.22 KiB)
# Int64 -> Int32
# 65.34 m
# workgroupsize=(16,16)
# 31.022 ms (35 allocations: 5.89 KiB)

function trace_image!(img, camera, scene)
    for xy in CartesianIndices(size(img))
        @_inbounds img[xy] = RGBf(trace_pixel(camera, scene, xy).c...)
    end
    return img
end

function threads_trace_image!(img, camera, bvh)
    Threads.@threads for xy in CartesianIndices(size(img))
        @_inbounds img[xy] = trace_pixel(camera, bvh, xy)
    end
    return img
end

@btime trace_image!(img, cam, bvh)
# Single: 707.754 ms (0 allocations: 0 bytes)
# New Triangle layout  1
# 860.535 ms (0 allocations: 0 bytes)
# GPU intersection compatible
# 403.335 ms (0 allocations: 0 bytes)

@btime threads_trace_image!(img, cam, bvh)
# Start
# Multi : 73.090 ms (262266 allocations: 156.04 MiB)
# BVH inline
# Multi (static): 66.564 ms (122 allocations: 45.62 KiB)
# New Triangle layout 1
# 80.222 ms (122 allocations: 32.88 KiB)
# 42.842 ms (122 allocations: 32.88 KiB) (more inline)
# GPU intersection compatible
# 42.681 ms (122 allocations: 32.88 KiB)


using Tullio

@_inbounds function tullio_trace_image!(img, camera, bvh)
    @tullio img[x, y] = trace_pixel(camera, bvh, (x, y))
    return img
end

@btime tullio_trace_image!(img, cam, bvh)
# BVH inline + tullio
# Multi: 150.944 ms (107 allocations: 33.17 KiB)
# New Triangle layout 1
# 161.447 ms (107 allocations: 33.17 KiB)
# 117.139 ms (107 allocations: 33.17 KiB) (more inline)
# GPU intersection compatible
# 82.461 ms (109 allocations: 22.39 KiB)

@btime launch_trace_image!(img, cam, bvh)
# 71.405 ms (233 allocations: 86.05 KiB)
# 47.240 ms (233 allocations: 86.09 KiB)
# GPU intersection compatible
# 44.629 ms (233 allocations: 54.50 KiB)


##########################
##########################
##########################
# Random benchmarks
v1 = Vec3f(0.0, 0.0, 0.0)
v2 = Vec3f(1.0, 0.0, 0.0)
v3 = Vec3f(0.0, 1.0, 0.0)

ray_origin = Vec3f(0.5, 0.5, 1.0)
ray_direction = Vec3f(0.0, 0.0, -1.0)

using Trace: Normal3f
m = Trace.create_triangle_mesh(Trace.ShapeCore(), UInt32[1, 2, 3], Point3f[v1, v2, v3], [Normal3f(0.0, 0.0, 1.0), Normal3f(0.0, 0.0, 1.0), Normal3f(0.0, 0.0, 1.0)])

t = Trace.Triangle(m, 1)
r = Trace.Ray(o=Point3f(ray_origin), d=ray_direction)
Trace.intersect_p(t, r)
Trace.intersect_triangle(r.o, r.d, t.vertices...)

# function launch_trace_image_ir!(img, camera, bvh, lights)
#     backend = KA.get_backend(img)
#     kernel! = ka_trace_image!(backend)
#     open("test2.ir", "w") do io
#         @device_code_llvm io begin
#             kernel!(img, camera, bvh, lights, ndrange = size(img), workgroupsize = (16, 16))
#         end
#     end
#     AMDGPU.synchronize(; stop_hostcalls=false)
#     return img
# end

ray = Trace.RayDifferentials(Trace.Ray(o=Point3f(0.5, 0.5, 1.0), d=Vec3f(0.0, 0.0, -1.0)))
open("li.llvm", "w") do io
    code_llvm(io, Trace.li, typeof.((Trace.UniformSampler(8), 5, ray, scene, 1)))
end

open("li-wt.jl", "w") do io
    code_warntype(io, Trace.li, typeof.((Trace.UniformSampler(8), 5, ray, scene, 1)))
end

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
