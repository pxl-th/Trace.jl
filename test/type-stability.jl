using Trace, GeometryBasics, StaticArrays

code_warntype(Trace._to_ray_coordinate_space, (SVector{3,Point3f}, Trace.Ray))
code_warntype(Trace.∂p, (Trace.Triangle, SVector{3,Point3f}, SVector{3,Point2f}))
code_warntype(Trace.∂n, (Trace.Triangle, SVector{3,Point2f}))
code_warntype(Trace.intersect, (Trace.Triangle, Trace.Ray, Bool))
code_warntype(Trace.intersect_triangle, (Trace.Triangle, Trace.Ray))
code_warntype(Trace.intersect_triangle, (Trace.Triangle, Trace.Ray))
code_warntype(Trace.intersect_p, (Trace.Triangle, Trace.Ray))

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
