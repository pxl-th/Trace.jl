# ============================================================================
# Scene generator for pbrt reference tests
#
# Generates .pbrt files covering:
#   - All material types × light types
#   - Roughness variants per material
#   - Texture variants (checkerboard reflectance)
#   - Filter variants (box, gaussian, mitchell, lanczos, triangle)
#   - Sensor variants (different ISO, white balance)
#   - Media (homogeneous cloud in dielectric sphere)
#
# Settings: 128x128, regularize=true, maxcomponentvalue=10
# Camera: zoomed in on sphere (dist=1.2, fov=40)
# ============================================================================

const SCENES_DIR = joinpath(@__DIR__, "scenes")
mkpath(SCENES_DIR)

# ============================================================================
# Scene defaults
# ============================================================================

const RES = 128
const CAMERA = """LookAt 0 -1.2 0.6   0 0 0.5   0 0 1
Camera "perspective" "float fov" 40"""
const FILM_BASE = """Film "rgb" "integer xresolution" $RES "integer yresolution" $RES "float maxcomponentvalue" 10"""
const INTEGRATOR = """Integrator "volpath" "integer maxdepth" 8 "bool regularize" true"""

# ============================================================================
# UV Sphere mesh generation with normals and UVs
# ============================================================================

function generate_uv_sphere(center=(0.0, 0.0, 0.5), radius=0.5;
                            n_lat=32, n_lon=64)
    vertices = Tuple{Float64,Float64,Float64}[]
    normals = Tuple{Float64,Float64,Float64}[]
    uvs = Tuple{Float64,Float64}[]
    indices = Tuple{Int,Int,Int}[]

    for i in 0:n_lat
        theta = pi * i / n_lat
        v = i / n_lat
        for j in 0:n_lon
            phi = 2pi * j / n_lon
            u = j / n_lon
            nx = sin(theta) * cos(phi)
            ny = sin(theta) * sin(phi)
            nz = cos(theta)
            push!(vertices, (center[1] + radius * nx, center[2] + radius * ny, center[3] + radius * nz))
            push!(normals, (nx, ny, nz))
            push!(uvs, (u, v))
        end
    end

    for i in 0:n_lat-1, j in 0:n_lon-1
        row = i * (n_lon + 1)
        next_row = (i + 1) * (n_lon + 1)
        push!(indices, (row + j, next_row + j, next_row + j + 1))
        push!(indices, (row + j, next_row + j + 1, row + j + 1))
    end

    return vertices, normals, uvs, indices
end

function generate_ground_disk(radius=2.0, n_segments=64)
    vertices = Tuple{Float64,Float64,Float64}[]
    normals = Tuple{Float64,Float64,Float64}[]
    uvs = Tuple{Float64,Float64}[]
    indices = Tuple{Int,Int,Int}[]

    push!(vertices, (0.0, 0.0, 0.0))
    push!(normals, (0.0, 0.0, 1.0))
    push!(uvs, (0.5, 0.5))

    for i in 0:n_segments
        angle = 2pi * i / n_segments
        push!(vertices, (radius * cos(angle), radius * sin(angle), 0.0))
        push!(normals, (0.0, 0.0, 1.0))
        push!(uvs, (0.5 + 0.5 * cos(angle), 0.5 + 0.5 * sin(angle)))
    end

    for i in 0:n_segments-1
        push!(indices, (0, i + 1, i + 2))
    end

    return vertices, normals, uvs, indices
end

# ============================================================================
# PBRT formatting
# ============================================================================

fmt_p3(pts) = join(["$(round(p[1],digits=6)) $(round(p[2],digits=6)) $(round(p[3],digits=6))" for p in pts], "  ")
fmt_p2(pts) = join(["$(round(p[1],digits=6)) $(round(p[2],digits=6))" for p in pts], "  ")
fmt_idx(tris) = join(["$(t[1]) $(t[2]) $(t[3])" for t in tris], "  ")

function write_mesh(io, verts, norms, uvs_arr, inds; indent="")
    println(io, "$(indent)Shape \"trianglemesh\"")
    println(io, "$(indent)  \"point3 P\" [ $(fmt_p3(verts)) ]")
    println(io, "$(indent)  \"normal N\" [ $(fmt_p3(norms)) ]")
    println(io, "$(indent)  \"point2 uv\" [ $(fmt_p2(uvs_arr)) ]")
    println(io, "$(indent)  \"integer indices\" [ $(fmt_idx(inds)) ]")
end

# Pre-generate meshes (shared across all scenes)
const SPHERE = generate_uv_sphere((0.0, 0.0, 0.5), 0.5; n_lat=32, n_lon=64)
const GROUND = generate_ground_disk(2.0, 64)

# ============================================================================
# Light definitions
# ============================================================================

const LIGHTS = Dict(
    "point"   => """LightSource "point" "rgb I" [40 40 40] "point3 from" [2 -1.5 3]""",
    "distant" => """LightSource "distant" "rgb L" [3.0 3.0 3.0] "point3 from" [2 -1 3] "point3 to" [0 0 0.5]""",
    "spot"    => """LightSource "spot" "rgb I" [80 80 80] "point3 from" [1.5 -1 2.5] "point3 to" [0 0 0.5] "float coneangle" 30 "float conedeltaangle" 5""",
    "area"    => """AttributeBegin
AreaLightSource "diffuse" "rgb L" [8 8 8] "bool twosided" true
Material "diffuse" "rgb reflectance" [1 1 1]
Translate 0.8 -0.4 1.5
Shape "trianglemesh"
  "point3 P" [ -0.4 -0.4 0  0.4 -0.4 0  0.4 0.4 0  -0.4 0.4 0 ]
  "integer indices" [ 0 1 2  0 2 3 ]
AttributeEnd""",
    "ambient" => """LightSource "infinite" "rgb L" [0.5 0.5 0.5]""",
)

# Extended light variants for light parameter tests
const LIGHT_VARIANTS = Dict(
    "point_warm"     => """LightSource "point" "rgb I" [50 30 10] "point3 from" [2 -1.5 3]""",
    "point_bright"   => """LightSource "point" "rgb I" [200 200 200] "point3 from" [2 -1.5 3]""",
    "spot_narrow"    => """LightSource "spot" "rgb I" [120 120 120] "point3 from" [1.5 -1 2.5] "point3 to" [0 0 0.5] "float coneangle" 15 "float conedeltaangle" 2""",
    "spot_wide"      => """LightSource "spot" "rgb I" [40 40 40] "point3 from" [1.5 -1 2.5] "point3 to" [0 0 0.5] "float coneangle" 60 "float conedeltaangle" 10""",
    "distant_colored" => """LightSource "distant" "rgb L" [2.0 0.5 0.5] "point3 from" [2 -1 3] "point3 to" [0 0 0.5]""",
)

# ============================================================================
# Material definitions
# ============================================================================

const MATERIALS = Dict(
    # Diffuse
    "diffuse"         => """Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]""",
    "diffuse_colored" => """Material "diffuse" "rgb reflectance" [0.8 0.2 0.1]""",

    # Conductor
    "conductor_gold"   => """Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0.01""",
    "conductor_mirror" => """Material "conductor" "float roughness" 0""",
    "conductor_rough"  => """Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0.2""",
    "conductor_silver" => """Material "conductor" "spectrum eta" "metal-Ag-eta" "spectrum k" "metal-Ag-k" "float roughness" 0.05""",
    "conductor_copper" => """Material "conductor" "spectrum eta" "metal-Cu-eta" "spectrum k" "metal-Cu-k" "float roughness" 0.1""",

    # Dielectric
    "dielectric"            => """Material "dielectric" "float eta" 1.5""",
    "dielectric_rough"      => """Material "dielectric" "float eta" 1.5 "float uroughness" 0.1 "float vroughness" 0.1""",
    "dielectric_rough_high" => """Material "dielectric" "float eta" 1.5 "float uroughness" 0.3 "float vroughness" 0.3""",
    "dielectric_diamond"    => """Material "dielectric" "float eta" 2.4""",

    # Thin dielectric
    "thindielectric" => """Material "thindielectric" "float eta" 1.5""",

    # Coated diffuse
    "coateddiffuse"       => """Material "coateddiffuse" "rgb reflectance" [0.5 0.3 0.1] "float roughness" 0.1 "float eta" 1.5""",
    "coateddiffuse_rough" => """Material "coateddiffuse" "rgb reflectance" [0.5 0.3 0.1] "float roughness" 0.3 "float eta" 1.5""",

    # Coated conductor
    "coatedconductor"       => """Material "coatedconductor" "spectrum conductor.eta" "metal-Au-eta" "spectrum conductor.k" "metal-Au-k" "float interface.roughness" 0.1 "float conductor.roughness" 0.01""",
    "coatedconductor_rough" => """Material "coatedconductor" "spectrum conductor.eta" "metal-Au-eta" "spectrum conductor.k" "metal-Au-k" "float interface.roughness" 0.3 "float conductor.roughness" 0.1""",

    # Diffuse transmission
    "diffusetransmission" => """Material "diffusetransmission" "rgb reflectance" [0.5 0.5 0.5] "rgb transmittance" [0.5 0.5 0.5]""",

    # Mix
    "mix" => """MakeNamedMaterial "mat_diffuse" "string type" "diffuse" "rgb reflectance" [0.8 0.1 0.1]
MakeNamedMaterial "mat_conductor" "string type" "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0.01
Material "mix" "string materials" [ "mat_diffuse" "mat_conductor" ] "float amount" 0.5""",
)

# Materials that use texture declarations (checkerboard reflectance)
const TEXTURE_MATERIALS = Dict(
    "diffuse_checker" => (
        textures = """Texture "checker_tex" "spectrum" "checkerboard"
  "rgb tex1" [0.8 0.2 0.1]
  "rgb tex2" [0.1 0.2 0.8]
  "float uscale" 8 "float vscale" 8""",
        material = """Material "diffuse" "texture reflectance" "checker_tex\""""
    ),
    "coateddiffuse_checker" => (
        textures = """Texture "checker_tex" "spectrum" "checkerboard"
  "rgb tex1" [0.6 0.4 0.2]
  "rgb tex2" [0.2 0.4 0.6]
  "float uscale" 8 "float vscale" 8""",
        material = """Material "coateddiffuse" "texture reflectance" "checker_tex" "float roughness" 0.1 "float eta" 1.5"""
    ),
    "conductor_checker_rough" => (
        textures = """Texture "rough_tex" "float" "checkerboard"
  "float tex1" 0.01
  "float tex2" 0.3
  "float uscale" 6 "float vscale" 6""",
        material = """Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "texture roughness" "rough_tex\""""
    ),
)

# Filter configurations
const FILTERS = Dict(
    "box"      => "",  # default (no filter directive = pbrt default)
    "gaussian" => """PixelFilter "gaussian" "float xradius" 1.5 "float yradius" 1.5 "float sigma" 0.5""",
    "mitchell" => """PixelFilter "mitchell" "float xradius" 2.0 "float yradius" 2.0""",
    "triangle" => """PixelFilter "triangle" "float xradius" 2.0 "float yradius" 2.0""",
    "lanczos"  => """PixelFilter "sinc" "float xradius" 4.0 "float yradius" 4.0 "float tau" 3.0""",
)

# Sensor configurations
const SENSORS = Dict(
    "default"   => "",  # default cie1931, iso=100
    "iso200"    => "iso200",   # just a tag — encoded in Film line
    "iso50"     => "iso50",
    "wb4000"    => "wb4000",   # warm white balance
    "wb8000"    => "wb8000",   # cool white balance
)

# ============================================================================
# Scene writing
# ============================================================================

function write_scene(filepath; camera=CAMERA, film=FILM_BASE, integrator=INTEGRATOR,
                     filter_line="", light="", textures="",
                     ground_mat="""Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]""",
                     sphere_mat="", medium_before="", medium_after="")
    open(filepath, "w") do io
        println(io, "# Auto-generated reference test scene")
        println(io, camera)
        println(io, film)
        !isempty(filter_line) && println(io, filter_line)
        println(io, integrator)
        println(io)
        println(io, "WorldBegin")
        println(io)

        # Medium definitions (before shapes)
        !isempty(medium_before) && (println(io, medium_before); println(io))

        # Light
        println(io, light)
        println(io)

        # Texture definitions
        !isempty(textures) && (println(io, textures); println(io))

        # Ground
        println(io, "# Ground")
        println(io, ground_mat)
        write_mesh(io, GROUND...)
        println(io)

        # Medium interface (if any)
        !isempty(medium_after) && println(io, medium_after)

        # Sphere
        println(io, "# Sphere")
        for line in split(sphere_mat, '\n')
            isempty(strip(line)) || println(io, line)
        end
        write_mesh(io, SPHERE...)
    end
    return filepath
end

# ============================================================================
# Generation functions
# ============================================================================

function generate_material_light_scenes()
    base_lights = ["point", "distant", "spot", "area", "ambient"]
    base_materials = [
        "diffuse", "diffuse_colored",
        "conductor_gold", "conductor_mirror", "conductor_rough",
        "conductor_silver", "conductor_copper",
        "dielectric", "dielectric_rough", "dielectric_rough_high", "dielectric_diamond",
        "thindielectric",
        "coateddiffuse", "coateddiffuse_rough",
        "coatedconductor", "coatedconductor_rough",
        "diffusetransmission",
    ]

    count = 0
    for mat in base_materials, light in base_lights
        name = "mat_$(mat)_light_$(light)"
        write_scene(joinpath(SCENES_DIR, "$(name).pbrt");
                    light=LIGHTS[light], sphere_mat=MATERIALS[mat])
        count += 1
    end

    # Mix (only point light)
    write_scene(joinpath(SCENES_DIR, "mat_mix_light_point.pbrt");
                light=LIGHTS["point"], sphere_mat=MATERIALS["mix"])
    count += 1

    println("  Generated $count material × light scenes")
    return count
end

function generate_texture_scenes()
    count = 0
    for (name, tex_mat) in TEXTURE_MATERIALS
        write_scene(joinpath(SCENES_DIR, "tex_$(name)_light_point.pbrt");
                    light=LIGHTS["point"], textures=tex_mat.textures,
                    sphere_mat=tex_mat.material)
        count += 1
    end
    println("  Generated $count texture scenes")
    return count
end

function generate_light_variant_scenes()
    count = 0
    mat = MATERIALS["diffuse"]
    for (lname, ldef) in LIGHT_VARIANTS
        write_scene(joinpath(SCENES_DIR, "light_$(lname).pbrt");
                    light=ldef, sphere_mat=mat)
        count += 1
    end
    println("  Generated $count light variant scenes")
    return count
end

function generate_filter_scenes()
    count = 0
    mat = MATERIALS["diffuse"]
    light = LIGHTS["point"]
    for (fname, fline) in FILTERS
        write_scene(joinpath(SCENES_DIR, "filter_$(fname).pbrt");
                    light=light, sphere_mat=mat, filter_line=fline)
        count += 1
    end
    println("  Generated $count filter scenes")
    return count
end

function generate_sensor_scenes()
    count = 0
    mat = MATERIALS["conductor_gold"]
    light = LIGHTS["point"]
    for (sname, _) in SENSORS
        sname == "default" && continue
        film = if sname == "iso200"
            """$FILM_BASE "float iso" 200"""
        elseif sname == "iso50"
            """$FILM_BASE "float iso" 50"""
        elseif sname == "wb4000"
            """$FILM_BASE "float whitebalance" 4000"""
        elseif sname == "wb8000"
            """$FILM_BASE "float whitebalance" 8000"""
        else
            FILM_BASE
        end
        write_scene(joinpath(SCENES_DIR, "sensor_$(sname).pbrt");
                    light=light, sphere_mat=mat, film=film)
        count += 1
    end
    println("  Generated $count sensor scenes")
    return count
end

function generate_medium_scenes()
    count = 0
    light = LIGHTS["point"]

    # Homogeneous medium in dielectric sphere (cloud-like)
    medium_def = """MakeNamedMedium "cloud"
  "string type" "homogeneous"
  "rgb sigma_a" [0.5 0.5 0.5]
  "rgb sigma_s" [10.0 10.0 10.0]
  "float g" 0.0
  "float scale" 1.0"""
    medium_interface = """MediumInterface "" "cloud\""""
    sphere_mat = """Material "dielectric" "float eta" 1.0"""

    write_scene(joinpath(SCENES_DIR, "medium_cloud_point.pbrt");
                light=light, sphere_mat=sphere_mat,
                medium_before=medium_def, medium_after=medium_interface)
    count += 1

    # Colored absorbing medium (tinted glass)
    medium_def2 = """MakeNamedMedium "tinted"
  "string type" "homogeneous"
  "rgb sigma_a" [0.1 0.5 0.1]
  "rgb sigma_s" [0.0 0.0 0.0]
  "float g" 0.0"""
    medium_interface2 = """MediumInterface "" "tinted\""""
    sphere_mat2 = """Material "dielectric" "float eta" 1.5"""

    write_scene(joinpath(SCENES_DIR, "medium_tinted_point.pbrt");
                light=light, sphere_mat=sphere_mat2,
                medium_before=medium_def2, medium_after=medium_interface2)
    count += 1

    println("  Generated $count medium scenes")
    return count
end

function generate_all_scenes()
    println("Generating scenes in $SCENES_DIR...")

    # Clear old scenes
    for f in readdir(SCENES_DIR)
        endswith(f, ".pbrt") && rm(joinpath(SCENES_DIR, f))
    end

    total = 0
    total += generate_material_light_scenes()
    total += generate_texture_scenes()
    total += generate_light_variant_scenes()
    total += generate_filter_scenes()
    total += generate_sensor_scenes()
    total += generate_medium_scenes()

    println("Total: $total scene files")
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all_scenes()
end
