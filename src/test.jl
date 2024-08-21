
x = Trace.scale(2, 2, 2)
x.inv_m == inv(x.m)
GLMakie.activate!(inline=true)
# Computer projective camera transformations.
resolution = Point2f(1024)
resolution = Trace.scale(resolution[1], resolution[2], 1)

window_width = screen_window.p_max .- screen_window.p_min
inv_bounds = Trace.scale((1.0f0 ./ window_width)..., 1)

offset = Trace.translate(Vec3f(
    -screen_window.p_min..., 0.0f0,
))

ray = camera.core.raster_to_camera(Point3f(1024 / 2, 1024 / 2, 0))
camera.core.screen_to_raster(camera.core.camera_to_screen(ray))

camera.core.screen_to_raster(Point3f(-1, -1, 0))
camera.core.screen_to_raster(Point3f(1, 1, 0))

camera.core.raster_to_screen(Point3f(1024, 1024, 0))

camera.core.screen_to_raster(Point3f(-1, -1, 0))# == (0, 0, 0)
camera.core.screen_to_raster(Point3f(1, 1, 0))# == (1024, 1024, 0)

camera.core.raster_to_screen(Point3f(0, 0, 0)) # == (0, 0, 0)
camera.core.screen_to_raster(Point3f(1, 1, 0)) # == (1024, 1024, 0)


res_t = Trace.scale(resolution..., 1)
window_width = screen_window.p_max .- screen_window.p_min
inv_bounds = Trace.scale((1.0f0 ./ window_width)..., 1)

offset = Trace.translate(Vec3f(
    (-screen_window.p_min)..., 0.0f0,
))

inv(res_t.m)
screen_to_raster = res_t * inv_bounds * offset

screen_to_raster(Point3f(1, 1, 0))

raster_to_screen = inv(offset) * inv(inv_bounds) * inv(res_t)

raster_to_screen(Point3f(1024, 1024, 0))

using GLMakie

ground = tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, -0.1)), mirror)
back = tmesh(Rect3f(Vec3f(-5, -3, 0), Vec3f(10, -0.1, 10)), material_white)
l = tmesh(Rect3f(Vec3f(-3, 0, 0), Vec3f(-0.1, 10, 10)), material_red)
r = tmesh(Rect3f(Vec3f(0, -3, 0), Vec3f(10, -0.1, 10)), material_blue)
begin
    s = Scene()
    cc = cam3d!(s)
    update_cam!(s, Point3f(0, 3, 3), Point3f(0, 0, 0), Vec3f(0, 0, 1))
    mesh!(s, Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, -0.1)), color=:white)
    mesh!(s, Rect3f(Vec3f(-5, -3, 0), Vec3f(10, -0.1, 10)), color=:blue)
    mesh!(s, Rect3f(Vec3f(-3, -5, 0), Vec3f(-0.1, 10, 10)), color=:red)
    mesh!(s, Rect3f(Vec3f(3, -5, 0), Vec3f(-0.1, 10, 10)), color=:green)
    s
end

cc.lookat[]
cc.eyeposition[]
cc.upvector[]
