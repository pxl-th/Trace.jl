import KernelAbstractions as KA

KA.@kernel some_kernel_f() = nothing

function some_kernel(arr)
    backend = KA.get_backend(arr)
    return some_kernel_f(backend)
end

function to_gpu(ArrayType, m::AbstractArray; preserve=[])
    arr = ArrayType(m)
    push!(preserve, arr)
    kernel = some_kernel(arr)
    return KA.argconvert(kernel, arr)
end

function to_gpu(ArrayType, m::Trace.Texture; preserve=[])
    @assert !Trace.no_texture(m)
    return Trace.Texture(
        to_gpu(ArrayType, m.data; preserve=preserve),
        m.const_value,
        m.isconst,
    )
end

function to_gpu(ArrayType, m::Trace.UberMaterial; preserve=[])
    if !Trace.no_texture(m.Kd)
        Kd = to_gpu(ArrayType, m.Kd; preserve=preserve)
        no_tex_s = typeof(Kd)()
        Kr = Trace.no_texture(m.Kr) ? no_tex_s : to_gpu(ArrayType, m.Kr; preserve=preserve)
    else
        Kr = to_gpu(ArrayType, m.Kr; preserve=preserve)
        no_tex_s = typeof(Kr)()
        Kd = Trace.no_texture(m.Kd) ? no_tex_s : to_gpu(ArrayType, m.Kd; preserve=preserve)
    end
    f_tex = to_gpu(ArrayType, Trace.Texture(zeros(Float32, 1, 1)); preserve=preserve)
    no_tex_f = typeof(f_tex)()
    return Trace.UberMaterial(
        Kd,
        Trace.no_texture(m.Ks) ? no_tex_s : to_gpu(ArrayType, m.Ks; preserve=preserve),
        Trace.no_texture(m.Kr) ? no_tex_s : to_gpu(ArrayType, m.Kr; preserve=preserve),
        Trace.no_texture(m.Kt) ? no_tex_s : to_gpu(ArrayType, m.Kt; preserve=preserve), Trace.no_texture(m.σ) ? no_tex_f : to_gpu(ArrayType, m.σ; preserve=preserve),
        Trace.no_texture(m.roughness) ? no_tex_f : to_gpu(ArrayType, m.roughness; preserve=preserve),
        Trace.no_texture(m.u_roughness) ? no_tex_f : to_gpu(ArrayType, m.u_roughness; preserve=preserve),
        Trace.no_texture(m.v_roughness) ? no_tex_f : to_gpu(ArrayType, m.v_roughness; preserve=preserve),
        Trace.no_texture(m.index) ? no_tex_f : to_gpu(ArrayType, m.index; preserve=preserve),
        m.remap_roughness,
        m.type,
    )
end

# Conversion constructor for e.g. GPU arrays
# TODO, create tree on GPU? Not sure if that will gain much though...
function to_gpu(ArrayType, bvh::Trace.BVHAccel; preserve=[])
    primitives = to_gpu(ArrayType, bvh.primitives; preserve=preserve)
    nodes = to_gpu(ArrayType, bvh.nodes; preserve=preserve)
    materials = to_gpu(ArrayType, to_gpu.((ArrayType,), bvh.materials; preserve=preserve); preserve=preserve)
    return Trace.BVHAccel(primitives, materials, bvh.max_node_primitives, nodes)
end

function to_gpu(ArrayType, scene::Trace.Scene; preserve=[])
    bvh = to_gpu(ArrayType, scene.aggregate; preserve=preserve)
    return Trace.Scene(scene.lights, bvh, scene.bound)
end

@kernel function ka_trace_image!(img, camera, scene, sampler, max_depth)
    _idx = @index(Global)
    idx = _idx % Int32
    @_inbounds if checkbounds(Bool, img, idx)
        cols = size(img, 2) % Int32
        row = (idx - Int32(1)) ÷ cols + Int32(1)
        col = (idx - Int32(1)) % cols + Int32(1)
        pixel = Point2f((row, cols - col))
        l = trace_pixel(camera, scene, pixel, sampler, max_depth)
        img[idx] = RGB{Float32}(( l.c)...)
    end
    nothing
end

function launch_trace_image!(img, camera, scene, samples_per_pixel::Int32, max_depth::Int32, niter::Int32)
    backend = KA.get_backend(img)
    kernel! = ka_trace_image!(backend)
    sampler = UniformSampler(samples_per_pixel)
    kernel!(img, camera, scene, sampler, max_depth, niter, ndrange=size(img), workgroupsize=(16, 16))
    KA.synchronize(backend)
    return img
end


function Trace.to_gpu(ArrayType, film::Film; preserve=[])
    return Film(
        film.resolution,
        film.crop_bounds,
        film.diagonal,
        KA.adapt(ArrayType, film.pixels),
        KA.adapt(ArrayType, film.tiles),
        film.tile_size,
        film.ntiles,
        film.filter_table,
        film.filter_table_width,
        film.filter_radius,
        film.scale,
        KA.adapt(ArrayType, film.framebuffer),
    )
end