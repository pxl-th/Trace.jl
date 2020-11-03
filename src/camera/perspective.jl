struct ProjectiveCamera <: Camera
    core::CameraCore
    camera_to_screen::Transformation
    raster_to_camera::Transformation
    screen_to_raster::Transformation
    raster_to_screen::Transformation

    lens_radius::Float32
    focal_distance::Float32

    function ProjectiveCamera(
        camera_to_world::Transformation, camera_to_screen::Transformation,
        screen_window::Bounds2,
        shutter_open::Float32, shutter_close::Float32,
        lens_radius::Float32, focal_distance::Float32,
        resolution::Tuple{Integer, Integer}, # TODO replace by film
    )
        core = CameraCore(camera_to_world, shutter_open, shutter_close)
        # Computer projective camera transformations.
        screen_to_raster = (
            scale(resolution[1], resolution[2], 1) *
            scale(
                1f0 / (screen_window.p_max[1] - screen_window.p_min[1]),
                1f0 / (screen_window.p_max[2] - screen_window.p_min[2]),
                1,
            ) *
            translate(Vec3f0(-screen_window.p_min[1], -screen_window.p_max[2], 0f0))
        )
        raster_to_screen = screen_to_raster |> inv
        raster_to_camera = inv(camera_to_screen) * raster_to_screen

        new(
            core,
            camera_to_screen, raster_to_camera,
            screen_to_raster, raster_to_screen,
            lens_radius, focal_distance,
        )
    end
end

struct PerspectiveCamera <: Camera
    core::ProjectiveCamera
    dx_camera::Vec3f0
    dy_camera::Vec3f0
    A::Float32

    function PerspectiveCamera(
        camera_to_world::Transformation, screen_window::Bounds2,
        shutter_open::Float32, shutter_close::Float32,
        lens_radius::Float32, focal_distance::Float32,
        fov::Float32,
        resolution::Tuple{Integer, Integer}, # TODO replace by film
    )
        pc = ProjectiveCamera(
            camera_to_world, perspective(fov, 0.01f0, 1000f0),
            screen_window, shutter_open, shutter_close,
            lens_radius, focal_distance, resolution,
        )

        p_min = pc.raster_to_camera(Point3f0(0))
        p_max = pc.raster_to_camera(Point3f0(resolution[1], resolution[2], 0f0))
        dx_camera = pc.raster_to_camera(Point3f0(1, 0, 0)) - p_min
        dy_camera = pc.raster_to_camera(Point3f0(0, 1, 0)) - p_min
        p = (p_min[1:2] ./ p_min[3]) - (p_max[1:2] ./ p_max[3])
        A = abs(p[1] * p[2])

        new(pc, dx_camera, dy_camera, A)
    end
end

function concentric_sample_disk(u::Point2f0)::Point2f0
    # Map uniform random numbers to [-1, 1].
    offset = 2f0 * u - Vec2f0(1f0)
    # Handle degeneracy at the origin.
    offset[1] ≈ 0 && offset[2] ≈ 0 && return Point2f0(0)
    if abs(offset[1]) > abs(offset[2])
        r = offset[1]
        θ = (offset[2] / offset[1]) * π / 4
    else
        r = offset[2]
        θ = π / 2 - (offset[1] / offset[2]) * π / 4
    end
    r * Point2f0(θ |> cos, θ |> sin)
end

function generate_ray(camera::PerspectiveCamera, sample::CameraSample)::Tuple{Ray, Float32}
    # Compute raster & camera sample positions.
    p_film = Point3f0(sample.film[1], sample.film[2], 0f0)
    p_camera = p_film |> camera.core.raster_to_camera

    ray = Ray(o=Point3f0(0), d=p_camera |> Vec3f0 |> normalize)
    # Modify ray for depth of field.
    if camera.core.lens_radius > 0
        # Sample points on lens.
        p_lens = camera.core.lens_radius * concentric_sample_disk(sample.lens)
        # Compute point on plane of focus.
        t = camera.core.focal_distance / ray.d[3]
        p_focus = t |> ray
        # Update ray for effects of lens.
        ray.o = Point3f0(p_lens[1], p_lens[2], 0f0)
        ray.d = normalize(Vec3f0(p_focus - ray.o))
    end

    ray.time = lerp(
        camera.core.core.shutter_open,
        camera.core.core.shutter_close,
        sample.time,
    )
    # TODO add medium
    ray = ray |> camera.core.core.camera_to_world
    ray, 1f0
end
