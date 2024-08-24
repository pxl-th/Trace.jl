using GeometryBasics, LinearAlgebra, Trace, BenchmarkTools
using ImageShow
using Makie
include("./../src/gpu-support.jl")

LowSphere(radius, contact=Point3f(0)) = Sphere(contact .+ Point3f(0, 0, radius), radius)

function tmesh(prim, material)

    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
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
    bvh = Trace.BVHAccel([s1, s2, s3, s4, ground, back, l, r])
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
        CUDA.@device_code_llvm io begin
            kernel!(img, camera, bvh, lights, ndrange=size(img), workgroupsize=(16, 16))
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
using CUDA
ArrayType = CuArray
preserve = []
gpu_bvh = to_gpu(ArrayType, bvh; preserve=preserve);
gpu_img = ArrayType(zeros(RGBf, res, res));
# launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(img, cam, bvh, lights);
# @btime launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
launch_trace_image!(gpu_img, cam, gpu_bvh, lights);
