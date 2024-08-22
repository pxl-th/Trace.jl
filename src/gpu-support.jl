to_gpu(ArrayType, m::AbstractArray) = AMDGPU.rocconvert(ArrayType(m))

function to_gpu(ArrayType, m::Trace.Texture)
    @assert !Trace.no_texture(m)
    return Trace.Texture(
        to_gpu(ArrayType, m.data),
        m.const_value,
        m.isconst,
    )
end

function to_gpu(ArrayType, m::Trace.UberMaterial)
    @assert !Trace.no_texture(m.Kd)
    Kd = to_gpu(ArrayType, m.Kd)
    no_tex_s = typeof(Kd)()
    f_tex = to_gpu(ArrayType, Trace.Texture(ArrayType(zeros(Float32, 1, 1))))
    no_tex_f = typeof(f_tex)()
    return Trace.UberMaterial(
        Kd,
        Trace.no_texture(m.Ks) ? no_tex_s : to_gpu(ArrayType, m.Ks),
        Trace.no_texture(m.Kr) ? no_tex_s : to_gpu(ArrayType, m.Kr),
        Trace.no_texture(m.Kt) ? no_tex_s : to_gpu(ArrayType, m.Kt),
        Trace.no_texture(m.σ) ? no_tex_f : to_gpu(ArrayType, m.σ),
        Trace.no_texture(m.roughness) ? no_tex_f : to_gpu(ArrayType, m.roughness),
        Trace.no_texture(m.u_roughness) ? no_tex_f : to_gpu(ArrayType, m.u_roughness),
        Trace.no_texture(m.v_roughness) ? no_tex_f : to_gpu(ArrayType, m.v_roughness),
        Trace.no_texture(m.index) ? no_tex_f : to_gpu(ArrayType, m.index),
        m.remap_roughness,
        m.type,
    )
end

# Conversion constructor for e.g. GPU arrays
# TODO, create tree on GPU? Not sure if that will gain much though...
function to_gpu(ArrayType, bvh::Trace.BVHAccel)
    primitives = to_gpu(ArrayType, bvh.primitives)
    nodes = to_gpu(ArrayType, bvh.nodes)
    materials = to_gpu(ArrayType, to_gpu.((ArrayType,), bvh.materials))
    return Trace.BVHAccel(primitives, materials, bvh.max_node_primitives, nodes)
end
