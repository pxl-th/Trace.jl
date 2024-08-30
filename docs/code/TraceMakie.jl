using Makie, Trace, ImageShow, Colors, FileIO, LinearAlgebra, GeometryBasics
using AMDGPU, GLMakie

ArrayType = ROCArray

function to_spectrum(data::Colorant)
    rgb = RGBf(data)
    return Trace.RGBSpectrum(rgb.r, rgb.g, rgb.b)
end

function to_spectrum(data::AbstractMatrix{<:Colorant})
    colors = convert(AbstractMatrix{RGBf}, data)
    return collect(reinterpret(Trace.RGBSpectrum, colors))
end

function extract_material(plot::Plot, tex::Union{Trace.Texture, Nothing})
    if haskey(plot, :material) && !isnothing(to_value(plot.material))
        if to_value(plot.material) isa Trace.Material
            return to_value(plot.material)
        end
    elseif tex isa Nothing
        error("Neither color nor material are defined for plot: $plot")
    else
        return Trace.MatteMaterial(tex, Trace.ConstantTexture(0.0f0))
    end
end

function extract_material(plot::Plot, color_obs::Observable)
    color = to_value(color_obs)
    tex = nothing
    if color isa AbstractMatrix{<:Number}
        calc_color = to_value(plot.calculated_colors)
        tex = Trace.Texture(to_spectrum(to_color(calc_color)))
        onany(plot, color_obs, plot.colormap, plot.colorrange) do color, cmap, crange
            tex.data = to_spectrum(to_color(calc_color))
            return
        end
    elseif color isa AbstractMatrix{<:Colorant}
        tex = Trace.Texture(to_spectrum(color))
        onany(plot, color_obs) do color
            tex.data = to_spectrum(color)
            return
        end
    elseif color isa Colorant || color isa Union{String,Symbol}
        tex = Trace.ConstantTexture(to_spectrum(color))
    elseif color isa Nothing
        # ignore!
        nothing
    else
        error("Unsupported color type for RadeonProRender backend: $(typeof(color))")
    end

    return extract_material(plot, tex)
end

function to_trace_primitive(plot::Makie.Mesh)
    # Potentially per instance attributes
    triangles = Trace.create_triangle_mesh(plot.mesh[])
    material = extract_material(plot, plot.color)
    return Trace.GeometricPrimitive(triangles, material)
end

function to_trace_primitive(plot::Makie.Surface)
    !plot.visible[] && return nothing
    x = plot[1]
    y = plot[2]
    z = plot[3]

    function grid(x, y, z, trans)
        space = to_value(get(plot, :space, :data))
        g = map(CartesianIndices(z)) do i
            p = Point3f(Makie.get_dim(x, i, 1, size(z)), Makie.get_dim(y, i, 2, size(z)), z[i])
            return Makie.apply_transform(trans, p, space)
        end
        return vec(g)
    end

    positions = lift(grid, x, y, z, Makie.transform_func_obs(plot))
    # normals = Makie.surface_normals(x[], y[], z[])
    r = Tesselation(Rect2f((0, 0), (1, 1)), size(z[]))
    # decomposing a rectangle into uv and triangles is what we need to map the z coordinates on
    # since the xyz data assumes the coordinates to have the same neighouring relations
    # like a grid
    faces = decompose(GLTriangleFace, r)
    uv = decompose_uv(r)
    # with this we can beuild a mesh
    mesh = normal_mesh(GeometryBasics.Mesh(meta(vec(positions[]), uv=uv), faces))

    triangles = Trace.create_triangle_mesh(mesh)
    material = extract_material(plot, plot.z)
    return Trace.GeometricPrimitive(triangles, material)
end

function to_trace_primitive(plot::Makie.Plot)
    return nothing
end

function to_trace_light(light::Makie.AmbientLight)
    return Trace.AmbientLight(
        to_spectrum(light.color[]),
    )
end
function to_trace_light(light::Makie.PointLight)
    return Trace.PointLight(
        Trace.translate(light.position[]), to_spectrum(light.color[]),
    )
end

function to_trace_light(light::Makie.PointLight)
    return Trace.PointLight(
        Trace.translate(light.position[]), to_spectrum(light.color[]),
    )
end

function to_trace_light(light::Makie.PointLight)
    return Trace.PointLight(
        Trace.translate(light.position[]), to_spectrum(light.color[]),
    )
end

function to_trace_light(light)
    return nothing
end

function to_trace_camera(scene::Makie.Scene, film)
    cc = scene.camera_controls
    return lift(scene, cc.eyeposition, cc.lookat, cc.upvector, cc.fov) do eyeposition, lookat, upvector, fov
        view = Trace.look_at(
            Point3f(eyeposition), Point3f(lookat), Vec3f(upvector),
        )
        return Trace.PerspectiveCamera(
            view, Trace.Bounds2(Point2f(-1.0f0), Point2f(1.0f0)),
            0.0f0, 1.0f0, 0.0f0, 1.0f6, Float32(fov),
            film
        )
    end
    return
end

function convert_scene(scene::Makie.Scene)
        # Only set background image if it isn't set by env light, since
    # background image takes precedence
    resolution = Point2f(size(scene))
    f = Trace.LanczosSincFilter(Point2f(1.0f0), 3.0f0)
    film = Trace.Film(
        resolution,
        Trace.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
        "trace-test.png",
    )
    primitives = []
    for plot in scene.plots
        prim = to_trace_primitive(plot)
        !isnothing(prim) && push!(primitives, prim)
    end
    camera = to_trace_camera(scene, film)
    lights = []
    for light in scene.lights
        l = to_trace_light(light)
        isnothing(l) || push!(lights, l)
    end
    if isempty(lights)
        error("Must have at least one light")
    end
    bvh = Trace.BVHAccel(primitives, 1)
    scene = Trace.Scene([lights...], bvh)
    return scene, camera, film
end

function render_whitted(mscene::Makie.Scene; samples_per_pixel=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    integrator = Trace.WhittedIntegrator(camera[], Trace.UniformSampler(samples_per_pixel), max_depth)
    integrator(scene, film)
    return reverse(film.framebuffer, dims=1)
end

function render_sppm(mscene::Makie.Scene; search_radius=0.075f0, max_depth=5, iterations=100)
    scene, camera, film = convert_scene(mscene)
    integrator = Trace.SPPMIntegrator(camera[], search_radius, max_depth, iterations, film)
    integrator(scene, film)
    return reverse(film.framebuffer, dims=1)
end

function render_gpu(mscene::Makie.Scene, ArrayType; samples_per_pixel=8, max_depth=5)
    scene, camera, film = convert_scene(mscene)
    preserve = []
    gpu_scene = Trace.to_gpu(ArrayType, scene; preserve=preserve)
    res = Int.((film.resolution...,))
    gpu_img = ArrayType(zeros(RGBf, res))
    GC.@preserve preserve begin
        Trace.launch_trace_image!(gpu_img, camera[], gpu_scene, Int32(samples_per_pixel), Int32(max_depth), Int32(0))
    end
    return Array(gpu_img)
end

function render(w::Whitten5, scene)

end

function render_interactive(mscene::Makie.Scene, ArrayType; max_depth=5)
    scene, camera, film = convert_scene(mscene)
    preserve = []
    gpu_scene = Trace.to_gpu(ArrayType, scene; preserve=preserve)
    res = Int.((film.resolution...,))
    gpu_img = ArrayType(zeros(RGBf, res))
    s = Scene(size=res)
    imgp = image!(s, -1..1, -1..1, Array(gpu_img))
    display(GLMakie.Screen(), mscene)
    display(GLMakie.Screen(), s)
    cam_start = camera[]
    n_iter = Int32(1)
    Base.errormonitor(@async while isopen(s)
        GC.@preserve preserve begin
            if cam_start != camera[]
                cam_start = camera[]
                gpu_img .= RGBf(0, 0, 0)
            end
            Trace.launch_trace_image!(gpu_img, camera[], gpu_scene, Int32(1), Int32(max_depth), n_iter)
            n_iter += Int32(1)
        end
        imgp[3] = Array(gpu_img)
        sleep(1/10)
    end)
    return Array(gpu_img)
end

begin
    catmesh = load(Makie.assetpath("cat.obj"))
    scene = Scene(size=(1024, 1024);
        lights=[AmbientLight(RGBf(0.7, 0.6, 0.6)), PointLight(Vec3f(0, 1, 0.5), RGBf(1.3, 1.3, 1.3))]
    )
    cam3d!(scene)
    mesh!(scene, catmesh, color=load(Makie.assetpath("diffusemap.png")))
    center!(scene)
    # @time render_whitted(scene)
    # 1.024328 seconds (16.94 M allocations: 5.108 GiB, 46.19% gc time, 81 lock conflicts)
    # 0.913530 seconds (16.93 M allocations: 5.108 GiB, 42.52% gc time, 57 lock conflicts)
    # 0.416158 seconds (75.58 k allocations: 88.646 MiB, 2.44% gc time, 16 lock conflicts)
    @time render_gpu(scene, ArrayType)
    # 0.135438 seconds (76.03 k allocations: 82.406 MiB, 8.57% gc time)
    # render_interactive(scene, ArrayType; max_depth=5)
end

begin
    scene = Scene(size=(1024, 1024);
        lights=[AmbientLight(RGBf(0.4, 0.4, 0.4)), PointLight(Vec3f(4, 4, 10), RGBf(500, 500, 500))]
    )
    cam3d!(scene)
    xs = LinRange(0, 10, 100)
    ys = LinRange(0, 15, 100)
    zs = [cos(x) * sin(y) for x in xs, y in ys]
    surface!(scene, xs, ys, zs)
    center!(scene)

    @time render_whitted(scene)
    # 1.598740s
    # 1.179450 seconds (17.30 M allocations: 5.126 GiB, 36.48% gc time, 94 lock conflicts)
    # 0.976180 seconds (443.12 k allocations: 107.841 MiB, 6.60% gc time, 12 lock conflicts)
    @time render_gpu(scene, ArrayType)
    # 0.236231 seconds (443.48 k allocations: 101.598 MiB, 3.92% gc time)
    # render_interactive(scene, ArrayType; max_depth=5)
end

begin
    model = load(joinpath(@__DIR__, "..", "src", "assets", "models", "caustic-glass.ply"))
    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.9f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(0.88f0)),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(1.4f0),
        true,
    )
    plastic = Trace.PlasticMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.5399999857f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
        Trace.ConstantTexture(0.010408001f0),
        true,
    )
    plastic_ceil = Trace.PlasticMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.3399999857f0, 0.6399999857f0, 0.8399999857f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1.4f0)),
        Trace.ConstantTexture(0.000408001f0),
        true,
    )
    scene = Scene(size=(1024, 1024); lights=[
        AmbientLight(RGBf(1, 1, 1)),
        PointLight(Vec3f(4, 4, 10), RGBf(150, 150, 150)),
        PointLight(Vec3f(-3, 10, 2.5), RGBf(60, 60, 60)),
        PointLight(Vec3f(0, 3, 0.5), RGBf(40, 40, 40))
    ])
    cam3d!(scene)
    cm = scene.camera_controls
    mesh!(scene, model, material=glass)
    mini, maxi = extrema(Rect3f(decompose(Point, model)))
    floorrect = Rect3f(Vec3f(-10, mini[2], -10), Vec3f(20, -1, 20))
    mesh!(scene, floorrect, material=plastic_ceil)
    ceiling = Rect3f(Vec3f(-25, 11, -25), Vec3f(50, -1, 50))
    mesh!(scene, ceiling, material=plastic)
    center!(scene)
    update_cam!(scene, Vec3f(-1.6, 6.2, 0.2), Vec3f(-3.6, 2.5, 2.4), Vec3f(0, 1, 0))

    @time render_whitted(scene)
    # 9.820304 seconds (1.69 M allocations: 235.165 MiB, 0.51% gc time, 3 lock conflicts)
    # @time render_gpu(scene, ArrayType)
    # 6.128600 seconds (1.70 M allocations: 228.875 MiB, 1.09% gc time)
    # @time render_sppm(scene; iterations=500)
    # @time colorbuffer(scene; backend=RPRMakie)
    # 6.321123 seconds (10.09 k allocations: 66.559 MiB, 0.15% gc time)
    # render_interactive(scene, ArrayType; max_depth=5)
end

@time begin
    tscene, tcamera, tfilm = convert_scene(scene)
    w = Whitten5(tfilm; samples_per_pixel=8)
end;
begin
    @time launch_trace_image!(w, tcamera[], tscene)
    Trace.to_framebuffer!(tfilm, 1.0f0)
    tfilm.framebuffer
end

# img = render_sppm(scene; iterations=1)
