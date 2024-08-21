using Makie, Trace, ImageShow, Colors, FileIO, LinearAlgebra, GeometryBasics

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
    triangles = Trace.create_triangle_mesh(
        plot.mesh[], Trace.ShapeCore(Trace.translate(Vec3f(0)), false),
    )
    material = extract_material(plot, plot.color)
    return [Trace.GeometricPrimitive(t, material) for t in triangles]
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
    normals = Makie.surface_normals(x[], y[], z[])
    r = Tesselation(Rect2f((0, 0), (1, 1)), size(z[]))
    # decomposing a rectangle into uv and triangles is what we need to map the z coordinates on
    # since the xyz data assumes the coordinates to have the same neighouring relations
    # like a grid
    faces = decompose(GLTriangleFace, r)
    uv = decompose_uv(r)
    # with this we can beuild a mesh
    mesh = normal_mesh(GeometryBasics.Mesh(meta(vec(positions[]), uv=uv), faces))

    triangles = Trace.create_triangle_mesh(
        mesh, Trace.ShapeCore(Trace.translate(Vec3f(0)), false),
    )
    material = extract_material(plot, plot.z)
    return [Trace.GeometricPrimitive(t, material) for t in triangles]
end
function to_trace_primitive(plot::Makie.Plot)
    return []
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
    fov = cc.fov[]
    view = Trace.look_at(
        Point3f(cc.eyeposition[]), Point3f(cc.lookat[]), Vec3f(cc.upvector[]),
    )
    return Trace.PerspectiveCamera(
        view, Trace.Bounds2(Point2f(-1.0f0), Point2f(1.0f0)),
        0.0f0, 1.0f0, 0.0f0, 1.0f6, Float32(fov),
        film
    )
end

function render_scene(scene::Makie.Scene)
    return render_scene(c -> Trace.WhittedIntegrator(c, Trace.UniformSampler(8), 5), scene)
end

function render_scene(integrator_f, mscene::Makie.Scene)
    # Only set background image if it isn't set by env light, since
    # background image takes precedence
    resolution = Point2f(size(mscene))
    f = Trace.LanczosSincFilter(Point2f(1.0f0), 3.0f0)
    film = Trace.Film(
        resolution,
        Trace.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
        "trace-test.png",
    )
    primitives = []
    for plot in mscene.plots
        prim = to_trace_primitive(plot)
        append!(primitives, prim)
    end
    camera = to_trace_camera(mscene, film)
    lights = []
    for light in mscene.lights
        l = to_trace_light(light)
        isnothing(l) || push!(lights, l)
    end
    if isempty(lights)
        error("Must have at least one light")
    end
    bvh = Trace.BVHAccel(map(identity, primitives), 1)
    integrator = integrator_f(camera)
    scene = Trace.Scene([lights...], bvh)
    integrator(scene)
    return reverse(film.framebuffer, dims=1)
end

catmesh = load(Makie.assetpath("cat.obj"))
begin
    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(1.25f0),
        true,
    )
    mirror = Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)))
    plastic = Trace.PlasticMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
        Trace.ConstantTexture(0.010408001f0),
        true,
    )
    scene = Scene(size=(1024, 1024); lights=[AmbientLight(RGBf(0.7, 0.6, 0.6)), PointLight(Vec3f(0, 1, 0.5), RGBf(1.3, 1.3, 1.3))])
    cam3d!(scene)
    mesh!(scene, catmesh, color=load(Makie.assetpath("diffusemap.png")))
    center!(scene)
    render_scene(scene)
end


begin
    scene = Scene(size=(1024, 1024); lights=[
        AmbientLight(RGBf(0.4, 0.4, 0.4)), PointLight(Vec3f(4, 4, 10), RGBf(500, 500, 500))])
    cam3d!(scene)
    xs = LinRange(0, 10, 100)
    ys = LinRange(0, 15, 100)
    zs = [cos(x) * sin(y) for x in xs, y in ys]
    surface!(scene, xs, ys, zs)
    center!(scene)
    render_scene(scene) do cam

    end
end
