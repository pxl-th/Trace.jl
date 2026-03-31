# ============================================================================
# PBRT scene builder — PBRTScene → Hikari Scene
# ============================================================================
# Converts the parsed intermediate representation into renderable Hikari objects.

const IDENTITY4 = Mat4f(LinearAlgebra.I)

struct PBRTResult
    scene::Scene
    camera::PerspectiveCamera
    film::Film
    integrator_settings::NamedTuple
end

"""
    load_pbrt(filename; backend=KA.CPU(), samples=nothing, max_depth=nothing)

Load a pbrt-v4 scene file and return a ready-to-render `PBRTResult`.

# Example
```julia
r = Hikari.load_pbrt("scene.pbrt")
img = VolPath(samples=64)(r.scene, r.film, r.camera)
```
"""
function load_pbrt(filename::AbstractString;
                   backend=KA.CPU(),
                   samples::Union{Nothing, Int}=nothing,
                   max_depth::Union{Nothing, Int}=nothing)
    pbrt = parse_pbrt(filename)
    build_hikari_scene(pbrt; backend=backend, samples=samples, max_depth=max_depth)
end

function build_hikari_scene(pbrt::PBRTScene;
                            backend=KA.CPU(),
                            samples::Union{Nothing, Int}=nothing,
                            max_depth::Union{Nothing, Int}=nothing)
    # --- Film ---
    xres = 512
    yres = 512
    if pbrt.film !== nothing
        xres = pbrt_get_int(pbrt.film, "xresolution", 512)
        yres = pbrt_get_int(pbrt.film, "yresolution", 512)
    end
    film = Film(Point2f(xres, yres))

    # --- Camera ---
    fov = 90f0
    if pbrt.camera !== nothing
        fov = Float32(pbrt_get_float(pbrt.camera, "fov", 90.0))
    end
    # pbrt's camera_transform is world-to-camera; we need camera-to-world
    cam_to_world = inv(pbrt.camera_transform)
    eye = Point3f(cam_to_world[1, 4], cam_to_world[2, 4], cam_to_world[3, 4])
    # pbrt looks along -Z in camera space → forward = -column3 of cam_to_world
    forward = normalize(-Vec3f(cam_to_world[1, 3], cam_to_world[2, 3], cam_to_world[3, 3]))
    target = Point3f(eye + forward)
    up = normalize(Vec3f(cam_to_world[1, 2], cam_to_world[2, 2], cam_to_world[3, 2]))
    camera = PerspectiveCamera(eye, target, film; up=up, fov=fov)

    # --- Integrator settings ---
    int_samples = 64
    int_max_depth = 8
    if pbrt.integrator !== nothing
        int_samples = pbrt_get_int(pbrt.integrator, "pixelsamples", 64)
        int_max_depth = pbrt_get_int(pbrt.integrator, "maxdepth", 8)
    end
    samples !== nothing && (int_samples = samples)
    max_depth !== nothing && (int_max_depth = max_depth)

    # --- Build materials cache ---
    mat_cache = Dict{String, Material}()
    for (name, entity) in pbrt.named_materials
        mat_cache[name] = build_pbrt_material(entity, pbrt)
    end

    # --- Build scene ---
    scene = Scene(; backend=backend)

    # Add standalone lights
    for lrec in pbrt.lights
        light = build_pbrt_light(lrec, pbrt)
        light !== nothing && push!(scene, light)
    end

    # Add shapes with materials
    for srec in pbrt.shapes
        mesh = build_pbrt_shape(srec, pbrt)
        mesh === nothing && continue
        mat = resolve_pbrt_material(srec, mat_cache, pbrt; scene=scene)

        # Area light → wrap material in MediumInterface with Emissive
        # pbrt normalizes: scale /= SpectrumToPhotometric(Lemit)
        if srec.area_light !== nothing
            Le = pbrt_get_rgb(srec.area_light, "L", (1.0, 1.0, 1.0))
            al_scale = Float32(pbrt_get_float(srec.area_light, "scale", 1.0))
            # Match pbrt's photometric normalization: scale /= SpectrumToPhotometric(Lemit)
            # Both pbrt and Hikari now use illuminant spectrum for area light emission,
            # so normalizing against the same illuminant spectrum gives matching results.
            table = get_srgb_table()
            Le_spectrum = rgb_illuminant_spectrum(table,
                RGB{Float32}(Float32(Le[1]), Float32(Le[2]), Float32(Le[3])))
            al_scale /= spectrum_to_photometric(Le_spectrum)
            emissive = Emissive(Le=Le, scale=al_scale)
            push!(scene, mesh, MediumInterface(emissive))
        else
            push!(scene, mesh, mat)
        end
    end

    sync!(scene)

    return PBRTResult(scene, camera, film,
        (samples=int_samples, max_depth=int_max_depth))
end

# ============================================================================
# Parameter extraction helpers
# ============================================================================

function pbrt_get_float(entity::PBRTEntity, name::String, default::Real)
    haskey(entity.params, name) || return default
    p = entity.params[name]
    isempty(p.values) && return default
    return Float64(p.values[1])
end

function pbrt_get_int(entity::PBRTEntity, name::String, default::Int)
    haskey(entity.params, name) || return default
    p = entity.params[name]
    isempty(p.values) && return default
    return Int(p.values[1])
end

function pbrt_get_string(entity::PBRTEntity, name::String, default::String)
    haskey(entity.params, name) || return default
    p = entity.params[name]
    isempty(p.values) && return default
    v = p.values[1]
    v isa AbstractString && return String(v)
    return default  # not a string value (e.g. spectrum stored as floats)
end

function pbrt_get_bool(entity::PBRTEntity, name::String, default::Bool)
    haskey(entity.params, name) || return default
    p = entity.params[name]
    isempty(p.values) && return default
    v = p.values[1]
    v isa Bool && return v
    v isa String && return lowercase(v) == "true"
    return default
end

function pbrt_get_rgb(entity::PBRTEntity, name::String, default::NTuple{3, Float64})
    haskey(entity.params, name) || return default
    p = entity.params[name]
    length(p.values) >= 3 || return default
    return (Float64(p.values[1]), Float64(p.values[2]), Float64(p.values[3]))
end

function pbrt_get_floats(entity::PBRTEntity, name::String)
    haskey(entity.params, name) || return Float64[]
    return Float64.(entity.params[name].values)
end

function pbrt_get_ints(entity::PBRTEntity, name::String)
    haskey(entity.params, name) || return Int[]
    return Int.(entity.params[name].values)
end

# ============================================================================
# Material building
# ============================================================================

function build_pbrt_material(entity::PBRTEntity, ::PBRTScene)
    type = lowercase(entity.type)

    if type == "diffuse"
        refl = pbrt_get_rgb(entity, "reflectance", (0.5, 0.5, 0.5))
        return Diffuse(Kd=refl)

    elseif type == "conductor"
        rough = Float32(pbrt_get_float(entity, "roughness", 0.0))
        urough = Float32(pbrt_get_float(entity, "uroughness", rough))
        vrough = Float32(pbrt_get_float(entity, "vroughness", rough))
        remap = pbrt_get_bool(entity, "remaproughness", true)
        # Check for named spectra (gold, silver, copper, etc.)
        eta_str = pbrt_get_string(entity, "eta", "")
        if contains(eta_str, "Au") || contains(eta_str, "gold")
            return Gold(roughness=max(urough, vrough))
        elseif contains(eta_str, "Ag") || contains(eta_str, "silver")
            return Silver(roughness=max(urough, vrough))
        elseif contains(eta_str, "Cu") && !contains(eta_str, "CuZn")
            return Copper(roughness=max(urough, vrough))
        elseif contains(eta_str, "Al")
            return Aluminum(roughness=max(urough, vrough))
        end
        # Generic conductor with reflectance
        refl = pbrt_get_rgb(entity, "reflectance", (1.0, 1.0, 1.0))
        return Conductor(reflectance=refl, roughness=max(urough, vrough),
                         remap_roughness=remap)

    elseif type == "dielectric"
        eta = Float32(pbrt_get_float(entity, "eta", 1.5))
        rough = Float32(pbrt_get_float(entity, "roughness", 0.0))
        remap = pbrt_get_bool(entity, "remaproughness", true)
        return Dielectric(index=eta, roughness=rough, remap_roughness=remap)

    elseif type == "thindielectric"
        eta = Float32(pbrt_get_float(entity, "eta", 1.5))
        return ThinDielectric(eta=eta)

    elseif type == "coateddiffuse"
        refl = pbrt_get_rgb(entity, "reflectance", (0.5, 0.5, 0.5))
        rough = Float32(pbrt_get_float(entity, "roughness", 0.0))
        eta = Float32(pbrt_get_float(entity, "eta", 1.5))
        remap = pbrt_get_bool(entity, "remaproughness", true)
        return CoatedDiffuse(reflectance=refl, roughness=rough, eta=eta,
                             remap_roughness=remap)

    elseif type == "coatedconductor"
        irough = Float32(pbrt_get_float(entity, "interface.roughness", 0.0))
        crough = Float32(pbrt_get_float(entity, "conductor.roughness", 0.0))
        ieta = Float32(pbrt_get_float(entity, "interface.eta", 1.5))
        ceta_str = pbrt_get_string(entity, "conductor.eta", "")
        # For named spectra, build a Conductor first, then extract its eta/k textures
        base = if contains(ceta_str, "Au") || contains(ceta_str, "gold")
            Gold(roughness=crough)
        elseif contains(ceta_str, "Ag") || contains(ceta_str, "silver")
            Silver(roughness=crough)
        elseif contains(ceta_str, "Cu") && !contains(ceta_str, "CuZn")
            Copper(roughness=crough)
        elseif contains(ceta_str, "Al")
            Aluminum(roughness=crough)
        else
            nothing
        end
        if base !== nothing
            # Use reflectance mode — spectral eta/k don't wrap in Texture easily
            return CoatedConductor(
                reflectance=base.reflectance,
                conductor_roughness=crough,
                interface_roughness=irough, interface_eta=ieta)
        end
        refl = pbrt_get_rgb(entity, "reflectance", (1.0, 1.0, 1.0))
        return CoatedConductor(reflectance=refl,
                               conductor_roughness=crough,
                               interface_roughness=irough,
                               interface_eta=ieta)

    elseif type == "diffusetransmission"
        refl = pbrt_get_rgb(entity, "reflectance", (0.25, 0.25, 0.25))
        trans = pbrt_get_rgb(entity, "transmittance", (0.25, 0.25, 0.25))
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        return DiffuseTransmission(reflectance=refl, transmittance=trans, scale=sc)

    elseif type == "mirror"
        refl = pbrt_get_rgb(entity, "reflectance", (0.9, 0.9, 0.9))
        return Mirror(Kr=refl)

    elseif type == "mix"
        # Mix needs special handling — returns nothing here,
        # handled in resolve_pbrt_mix_material with scene context
        return nothing

    else
        @warn "pbrt: unsupported material type '$type', using default Diffuse"
        return Diffuse(Kd=(0.5, 0.5, 0.5))
    end
end

"""Build a MixMaterial, resolving sub-material references from named materials cache."""
function build_pbrt_mix_material(entity::PBRTEntity, mat_cache::Dict{String, Material},
                                 scene::Scene, ::PBRTScene)
    # Get sub-material names from "string materials" param
    haskey(entity.params, "materials") || error("mix material without 'materials' param")
    mat_names = String.(entity.params["materials"].values)
    length(mat_names) == 2 || error("mix material needs exactly 2 sub-materials, got $(length(mat_names))")

    amount = Float32(pbrt_get_float(entity, "amount", 0.5))

    # Resolve sub-materials
    mat1 = get(mat_cache, mat_names[1], nothing)
    mat2 = get(mat_cache, mat_names[2], nothing)
    mat1 === nothing && error("mix: unknown material '$(mat_names[1])'")
    mat2 === nothing && error("mix: unknown material '$(mat_names[2])'")

    # Push sub-materials to scene.materials to get SetKeys
    key1 = push!(scene.materials, mat1)
    key2 = push!(scene.materials, mat2)

    return MixMaterial(mat1, mat2, to_texture(amount), key1, key2)
end

function resolve_pbrt_material(srec::PBRTShapeRecord, mat_cache::Dict{String, Material},
                               pbrt::PBRTScene; scene::Union{Scene, Nothing}=nothing)
    entity = if srec.material_name !== nothing
        name = srec.material_name
        if haskey(mat_cache, name)
            return mat_cache[name]
        elseif haskey(pbrt.named_materials, name)
            pbrt.named_materials[name]
        else
            @warn "pbrt: unknown named material '$name', using default"
            return Diffuse(Kd=(0.5, 0.5, 0.5))
        end
    elseif srec.material_inline !== nothing
        srec.material_inline
    else
        return Diffuse(Kd=(0.5, 0.5, 0.5))
    end

    # Handle mix material specially (needs scene for SetKey assignment)
    if lowercase(entity.type) == "mix" && scene !== nothing
        return build_pbrt_mix_material(entity, mat_cache, scene, pbrt)
    end

    mat = build_pbrt_material(entity, pbrt)
    mat !== nothing && return mat
    return Diffuse(Kd=(0.5, 0.5, 0.5))
end

# ============================================================================
# Shape building
# ============================================================================

function build_pbrt_shape(srec::PBRTShapeRecord, pbrt::PBRTScene)
    entity = srec.entity
    type = lowercase(entity.type)

    if type == "sphere"
        radius = Float32(pbrt_get_float(entity, "radius", 1.0))
        mesh = tessellate_sphere(radius; segments=64)
        return apply_pbrt_transform(mesh, srec.transform)

    elseif type == "disk"
        radius = Float32(pbrt_get_float(entity, "radius", 1.0))
        mesh = tessellate_disk(radius; segments=64)
        return apply_pbrt_transform(mesh, srec.transform)

    elseif type == "trianglemesh"
        return build_trianglemesh(entity, srec.transform)

    elseif type == "plymesh"
        filename = pbrt_get_string(entity, "filename", "")
        isempty(filename) && (@warn "pbrt: plymesh without filename"; return nothing)
        path = isabspath(filename) ? filename : joinpath(pbrt.base_dir, filename)
        isfile(path) || (@warn "pbrt: PLY file not found: $path"; return nothing)
        mesh = FileIO.load(path)
        return apply_pbrt_transform(mesh, srec.transform)

    elseif type == "loopsubdiv"
        return build_trianglemesh(entity, srec.transform)

    else
        @warn "pbrt: unsupported shape type '$type'"
        return nothing
    end
end

function build_trianglemesh(entity::PBRTEntity, transform::Mat4f)
    P = pbrt_get_floats(entity, "P")
    indices = pbrt_get_ints(entity, "indices")
    (isempty(P) || isempty(indices)) && return nothing

    n_verts = length(P) ÷ 3
    points = [Point3f(P[3i-2], P[3i-1], P[3i]) for i in 1:n_verts]

    # Apply transform
    if transform != IDENTITY4
        for i in eachindex(points)
            p = points[i]
            p4 = transform * Vec4f(p[1], p[2], p[3], 1f0)
            points[i] = Point3f(p4[1] / p4[4], p4[2] / p4[4], p4[3] / p4[4])
        end
    end

    n_tris = length(indices) ÷ 3
    # pbrt uses 0-based indices
    faces = [TriangleFace{Int}(indices[3i-2]+1, indices[3i-1]+1, indices[3i]+1)
             for i in 1:n_tris]

    # Optional normals
    N = pbrt_get_floats(entity, "N")
    if !isempty(N) && length(N) == 3 * n_verts
        normals = [Normal3f(N[3i-2], N[3i-1], N[3i]) for i in 1:n_verts]
        return GeometryBasics.Mesh(
            GeometryBasics.meta(points; normals=normals), faces)
    end

    # Optional UVs
    uv = pbrt_get_floats(entity, "uv")
    if !isempty(uv) && length(uv) == 2 * n_verts
        uvs = [Point2f(uv[2i-1], uv[2i]) for i in 1:n_verts]
        return GeometryBasics.Mesh(
            GeometryBasics.meta(points; uv=uvs), faces)
    end

    return GeometryBasics.Mesh(points, faces)
end

function apply_pbrt_transform(mesh, transform::Mat4f)
    transform == IDENTITY4 && return mesh

    positions = GeometryBasics.coordinates(mesh)
    faces = GeometryBasics.faces(mesh)
    new_positions = map(positions) do p
        p4 = transform * Vec4f(p[1], p[2], p[3], 1f0)
        Point3f(p4[1] / p4[4], p4[2] / p4[4], p4[3] / p4[4])
    end
    return GeometryBasics.Mesh(collect(new_positions), collect(faces))
end

# UV sphere tessellation
function tessellate_sphere(radius::Float32; segments::Int=64)
    rings = segments ÷ 2
    points = Point3f[]
    faces = TriangleFace{Int}[]

    push!(points, Point3f(0f0, 0f0, radius))

    for i in 1:rings-1
        theta = Float32(π) * i / rings
        st, ct = sincos(theta)
        for j in 1:segments
            phi = 2f0 * Float32(π) * (j - 1) / segments
            sp, cp = sincos(phi)
            push!(points, Point3f(radius * st * cp, radius * st * sp, radius * ct))
        end
    end

    push!(points, Point3f(0f0, 0f0, -radius))

    for j in 1:segments
        j_next = mod1(j + 1, segments)
        push!(faces, TriangleFace{Int}(1, 1 + j, 1 + j_next))
    end

    for i in 1:rings-2
        for j in 1:segments
            j_next = mod1(j + 1, segments)
            a = 1 + (i - 1) * segments + j
            b = 1 + (i - 1) * segments + j_next
            c = 1 + i * segments + j
            d = 1 + i * segments + j_next
            push!(faces, TriangleFace{Int}(a, c, b))
            push!(faces, TriangleFace{Int}(b, c, d))
        end
    end

    bottom = length(points)
    base = 1 + (rings - 2) * segments
    for j in 1:segments
        j_next = mod1(j + 1, segments)
        push!(faces, TriangleFace{Int}(bottom, base + j_next, base + j))
    end

    return GeometryBasics.Mesh(points, faces)
end

function tessellate_disk(radius::Float32; segments::Int=64)
    points = Point3f[Point3f(0f0, 0f0, 0f0)]
    faces = TriangleFace{Int}[]

    for j in 1:segments
        phi = 2f0 * Float32(π) * (j - 1) / segments
        sp, cp = sincos(phi)
        push!(points, Point3f(radius * cp, radius * sp, 0f0))
    end

    for j in 1:segments
        j_next = mod1(j + 1, segments)
        push!(faces, TriangleFace{Int}(1, 1 + j, 1 + j_next))
    end

    return GeometryBasics.Mesh(points, faces)
end

# ============================================================================
# Light building
# ============================================================================

function build_pbrt_light(lrec::PBRTLightRecord, pbrt::PBRTScene)
    entity = lrec.entity
    type = lowercase(entity.type)

    if type == "point"
        rgb = pbrt_get_rgb(entity, "I", (1.0, 1.0, 1.0))
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        from = pbrt_get_rgb(entity, "from", (0.0, 0.0, 0.0))
        pos = Vec3f(Float32(from[1]), Float32(from[2]), Float32(from[3]))
        if lrec.transform != IDENTITY4
            p4 = lrec.transform * Vec4f(pos[1], pos[2], pos[3], 1f0)
            pos = Vec3f(p4[1] / p4[4], p4[2] / p4[4], p4[3] / p4[4])
        end
        # Use RGB constructor (not RGBSpectrum) — matches Hikari's expected intensity model
        return PointLight(
            RGB{Float32}(Float32(rgb[1]) * sc, Float32(rgb[2]) * sc, Float32(rgb[3]) * sc),
            pos)

    elseif type == "distant"
        rgb = pbrt_get_rgb(entity, "L", (1.0, 1.0, 1.0))
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        from = pbrt_get_rgb(entity, "from", (0.0, 0.0, 0.0))
        to = pbrt_get_rgb(entity, "to", (0.0, 0.0, 1.0))
        dir = Vec3f(Float32(to[1] - from[1]), Float32(to[2] - from[2]),
                    Float32(to[3] - from[3]))
        return DirectionalLight(
            RGB{Float32}(Float32(rgb[1]) * sc, Float32(rgb[2]) * sc, Float32(rgb[3]) * sc),
            dir)

    elseif type == "spot"
        rgb = pbrt_get_rgb(entity, "I", (1.0, 1.0, 1.0))
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        cone = Float32(pbrt_get_float(entity, "coneangle", 30.0))
        delta = Float32(pbrt_get_float(entity, "conedeltaangle", 5.0))
        from = pbrt_get_rgb(entity, "from", (0.0, 0.0, 0.0))
        pos = Point3f(Float32(from[1]), Float32(from[2]), Float32(from[3]))
        to = pbrt_get_rgb(entity, "to", (0.0, 0.0, 1.0))
        target = Point3f(Float32(to[1]), Float32(to[2]), Float32(to[3]))
        # SpotLight(rgb, position, target, total_width_deg, falloff_start_deg)
        return SpotLight(
            RGB{Float32}(Float32(rgb[1]) * sc, Float32(rgb[2]) * sc, Float32(rgb[3]) * sc),
            pos, target, cone, cone - delta)

    elseif type == "infinite"
        filename = pbrt_get_string(entity, "filename", "")
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        if !isempty(filename)
            path = isabspath(filename) ? filename : joinpath(pbrt.base_dir, filename)
            if isfile(path)
                return EnvironmentLight(path; scale=RGBSpectrum(sc))
            end
        end
        rgb = pbrt_get_rgb(entity, "L", (1.0, 1.0, 1.0))
        return AmbientLight(
            RGB{Float32}(Float32(rgb[1]) * sc, Float32(rgb[2]) * sc, Float32(rgb[3]) * sc))

    else
        @warn "pbrt: unsupported light type '$type'"
        return nothing
    end
end

# ============================================================================
# Convenience render function
# ============================================================================

"""
    render_pbrt(filename; backend, samples, max_depth, output)

Load a pbrt scene, render it, and optionally save to file. Returns the rendered image.
"""
function render_pbrt(filename::AbstractString;
                     backend=KA.CPU(),
                     samples::Union{Nothing, Int}=nothing,
                     max_depth::Union{Nothing, Int}=nothing,
                     output::Union{Nothing, String}=nothing)
    r = load_pbrt(filename; backend=backend, samples=samples, max_depth=max_depth)
    vp = VolPath(samples=r.integrator_settings.samples,
                 max_depth=r.integrator_settings.max_depth)
    img = vp(r.scene, r.film, r.camera)
    if output !== nothing
        FileIO.save(output, img)
    end
    return img
end
