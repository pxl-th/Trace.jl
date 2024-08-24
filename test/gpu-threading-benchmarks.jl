using GeometryBasics, LinearAlgebra, Trace, BenchmarkTools
using ImageShow
using Makie
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

using KernelAbstractions.Extras.LoopInfo: @unroll

function simple_shading(bvh, shape, ray, si, l, depth, max_depth, lights)
    core = si.core
    n = si.shading.n
    wo = core.wo
    # Compute scattering functions for surface interaction.
    si = Trace.compute_differentials(si, ray)
    mat = Trace.get_material(bvh, shape)
    if mat.type === Trace.NO_MATERIAL
        return l
    end
    bsdf = mat(si, false, Trace.Radiance)
    # Compute emitted light if ray hit an area light source.
    l += Trace.le(si, wo)
    # Add contribution of each light source.
    @unroll for light in lights
        sampled_li, wi, pdf, vt = Trace.sample_li(
            light, core, rand(Point2f),
        )
        (Trace.is_black(sampled_li) || pdf ≈ 0.0f0) && continue
        f = bsdf(wo, wi)
        if !Trace.is_black(f) && !Trace.intersect_p(bvh, Trace.spawn_ray(vt.p0, vt.p1))
            l += f * sampled_li * abs(wi ⋅ n) / pdf
        end
    end
    # if depth + 1 <= max_depth
    #     # Trace rays for specular reflection & refraction.
    #     l += specular_reflect(bsdf, i, ray, si, scene, depth)
    #     l += specular_transmit(bsdf, i, ray, si, scene, depth)
    # end
    return l
end


@inline function trace_pixel(camera, bvh, xy, lights)
    pixel = Point2f(Tuple(xy))
    camera_sample = get_camera_sample(pixel)
    ray, ω = Trace.generate_ray_differential(camera, camera_sample)
    l = Trace.RGBSpectrum(0.0f0)
    if ω > 0.0f0
        hit, shape, si = Trace.intersect!(bvh, ray)
        if hit
            l = simple_shading(bvh, shape, ray, si, l, 1, 8, lights)
        end
    end
    return RGBf(l.c...)
end

using KernelAbstractions
import KernelAbstractions as KA


@kernel function ka_trace_image!(img, camera, bvh, lights)
    idx = @index(Global, Linear)
    if checkbounds(Bool, img, idx)
        xy = Tuple(divrem(idx, size(img, 1)))
        @inbounds img[idx] = trace_pixel(camera, bvh, xy, lights)
    end
end

function launch_trace_image_ir!(img, camera, bvh, lights)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    open("test2.ir", "w") do io
        @device_code_llvm io begin
            kernel!(img, camera, bvh, lights, ndrange = size(img), workgroupsize = (16, 16))
        end
    end
    AMDGPU.synchronize(; stop_hostcalls=false)
    return img
end
function launch_trace_image!(img, camera, bvh, lights)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    kernel!(img, camera, bvh, lights, ndrange=size(img), workgroupsize=(16, 16))
    KA.synchronize(backend)
    return img
end
# using AMDGPU
# ArrayType = ROCArray
# using CUDA
# ArrayType = CuArray

using Metal
ArrayType = MtlArray
preserve = []
gpu_bvh = to_gpu(ArrayType, bvh; preserve=preserve);
gpu_img = ArrayType(zeros(RGBf, res, res));
# launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
@btime launch_trace_image!(img, cam, bvh, lights)
# 76.420 ms (234 allocations: 86.05 KiB)
Array(gpu_img)

function cu_trace_image!(img, camera, bvh, lights)
    x = threadIdx().x
    y = threadIdx().y
    if checkbounds(Bool, img, (x, y))
        @inbounds img[x, y] = trace_pixel(camera, bvh, (x,y), lights)
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

function trace_image!(img, camera, bvh)
    for xy in CartesianIndices(size(img))
        @inbounds img[xy] = trace_pixel(camera, bvh, xy)
    end
    return img
end

function threads_trace_image!(img, camera, bvh)
    Threads.@threads :static for xy in CartesianIndices(size(img))
        @inbounds img[xy] = trace_pixel(camera, bvh, xy)
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

@inbounds function tullio_trace_image!(img, camera, bvh)
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
