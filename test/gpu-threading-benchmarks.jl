using GeometryBasics, LinearAlgebra, Trace, BenchmarkTools, AMDGPU
using FileIO
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
    img = zeros(RGBf, res, res)
end

@inline function get_camera_sample(p_raster::Point2)

    p_film = p_raster .+ rand(Point2f)
    p_lens = rand(Point2f)
    Trace.CameraSample(p_film, p_lens, rand(Float32))
end

@inline function trace_pixel(camera, bvh, xy)

    pixel = Point2f(Tuple(xy))
    camera_sample = get_camera_sample(pixel)
    ray, ω = Trace.generate_ray_differential(camera, camera_sample)
    hit, primitive, interaction = Trace.intersect!(bvh, ray)
    return ifelse(hit, RGBf(interaction.core.n...), RGBf(0.0f0, 0.0f0, 0.0f0))
end

using KernelAbstractions
import KernelAbstractions as KA


@kernel function ka_trace_image!(img, camera, bvh)
    xy = @index(Global, Cartesian)
    @inbounds img[xy] = trace_pixel(camera, bvh, xy)
end

function launch_trace_image!(img, camera, bvh)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    kernel!(img, camera, bvh, ndrange=size(img))
    KA.synchronize(backend)
    return img
end

gpu_bvh = to_gpu(ROCArray, bvh);
gpu_img = ROCArray(zeros(RGBf, res, res));
@btime launch_trace_image!(gpu_img, cam, gpu_bvh);
Array(gpu_img)
# 380.081 ms (913 allocations: 23.55 KiB)

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
