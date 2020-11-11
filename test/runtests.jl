using Test
using GeometryBasics
using LinearAlgebra
using Trace

@testset "Ray-Bounds intersection" begin
    b = Trace.Bounds3(Point3f0(1), Point3f0(2))
    b_neg = Trace.Bounds3(Point3f0(-2), Point3f0(-1))
    r0 = Trace.Ray(o=Point3f0(0), d=Vec3f0(1, 0, 0))
    r1 = Trace.Ray(o=Point3f0(0), d=Vec3f0(1))
    ri = Trace.Ray(o=Point3f0(1.5), d=Vec3f0(1, 1, 0))

    r, t0, t1 = Trace.intersect_p(b, r1)
    @test r && t0 ≈ 1f0 && t1 ≈ 2f0
    r, t0, t1 = Trace.intersect_p(b, r0)
    @test !r && t0 ≈ 0f0 && t1 ≈ 0f0
    r, t0, t1 = Trace.intersect_p(b, ri)
    @test r && t0 ≈ 0f0 && t1 ≈ 0.5f0

    # Test intersection with precomputed direction reciprocal.
    inv_dir = 1f0 ./ r1.d
    dir_is_negative = r1.d |> Trace.is_dir_negative
    @test Trace.intersect_p(b, r1, inv_dir, dir_is_negative)
    @test !Trace.intersect_p(b_neg, r1, inv_dir, dir_is_negative)
end

@testset "Sphere bound" begin
    core = Trace.ShapeCore(
        Trace.translate(Vec3f0(0)), Trace.translate(Vec3f0(0)), false,
    )
    s = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)

    sb = s |> Trace.object_bound
    @test sb[1] == Point3f0(-1f0)
    @test sb[2] == Point3f0(1f0)
end

@testset "Ray-Sphere insersection" begin
    # Sphere at the origin.
    core = Trace.ShapeCore(
        Trace.translate(Vec3f0(0)), Trace.translate(Vec3f0(0)), false,
    )
    s = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)
    r = Trace.Ray(o=Point3f0(0, -2, 0), d=Vec3f0(0, 1, 0))

    i, t, interaction = Trace.intersect(s, r, false)
    ip = Trace.intersect_p(s, r, false)

    @test i == ip
    @test t ≈ 1f0
    @test r(t) ≈ Point3f0(0, -1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f0(0, -1, 0) # Object intersection.

    # Translated sphere.
    core = Trace.ShapeCore(
        Trace.translate(Vec3f0(0, 2, 0)), Trace.translate(Vec3f0(0, -2, 0)),
        false,
    )
    s = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)
    r = Trace.Ray(o=Point3f0(0, 0, 0), d=Vec3f0(0, 1, 0))

    i, t, interaction = Trace.intersect(s, r, false)
    ip = Trace.intersect_p(s, r, false)

    @test i == ip
    @test t ≈ 1f0
    @test r(t) ≈ Point3f0(0, 1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f0(0, -1, 0) # Object intesection.
end

@testset "Test triangle" begin
    core = Trace.ShapeCore(Trace.translate(Vec3f0(0, 0, 2)), Trace.translate(Vec3f0(0, 0, -2)), false)
    triangles = Trace.create_triangle_mesh(
        core,
        1, UInt32[1, 2, 3],
        3, [Point3f0(0, 0, 0), Point3f0(1, 0, 0), Point3f0(1, 1, 0)],
        [Trace.Normal3f0(0, 0, -1), Trace.Normal3f0(0, 0, -1), Trace.Normal3f0(0, 0, -1)],
    )

    tv = triangles[1] |> Trace.vertices
    a = norm(tv[1] - tv[2]) ^ 2 * 0.5f0
    @test Trace.area(triangles[1]) ≈ a

    target_wb = Trace.Bounds3(Point3f0(0, 0, 2), Point3f0(1, 1, 2))
    target_ob = Trace.Bounds3(Point3f0(0, 0, 0), Point3f0(1, 1, 0))
    @test Trace.object_bound(triangles[1]) ≈ target_ob
    @test Trace.world_bound(triangles[1]) ≈ target_wb

    ray = Trace.Ray(o=Point3f0(0, 0, -2), d=Vec3f0(0, 0, 1))
    intersects, t_hit, interaction = Trace.intersect(triangles[1], ray)
    # CS origin is at the ray's origin, and triangle transformed to that CS.
    target_intersection = Point3f0(0, 0, 4)
    target_uv = Point2f0(0)

    @test intersects
    @test t_hit ≈ 4f0
    @test interaction.core.p ≈ target_intersection
    @test interaction.core.wo ≈ -ray.d
    @test interaction.uv ≈ target_uv

    ray = Trace.Ray(o=Point3f0(1, 0.5, 0), d=Vec3f0(0, 0, 1))
    intersects, t_hit, interaction = Trace.intersect(triangles[1], ray)
    # CS origin is at the ray's origin, and triangle transformed to that CS.
    target_intersection = Point3f0(0, 0, 2)
    target_uv = Point2f0(1, 0.5)

    @test intersects
    @test t_hit ≈ 2f0
    @test interaction.core.p ≈ target_intersection
    @test interaction.core.wo ≈ -ray.d
    @test interaction.uv ≈ target_uv
end

@testset "BVH SAH" begin
    primitives = Trace.Primitive[]
    for i in 0:3:21
        core = Trace.ShapeCore(
            Trace.translate(Vec3f0(i, i, 0)),
            Trace.translate(Vec3f0(-i, -i, 0)), false,
        )
        sphere = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)
        push!(primitives, Trace.GeometricPrimitive(sphere))
    end

    bvh = Trace.BVHAccel{Trace.SAH}(primitives[1:4])
    bvh2 = Trace.BVHAccel{Trace.SAH}(Trace.Primitive[primitives[5:end]..., bvh])
    @test Trace.world_bound(bvh) ≈ Trace.Bounds3(Point3f0(-1f0), Point3f0(10f0, 10f0, 1f0))
    @test Trace.world_bound(bvh2) ≈ Trace.Bounds3(Point3f0(-1f0), Point3f0(22f0, 22f0, 1f0))

    ray1 = Trace.Ray(o=Point3f0(-2f0, 0f0, 0f0), d=Vec3f0(1f0, 0f0, 0f0))
    ray2 = Trace.Ray(o=Point3f0(0f0, 18f0, 0f0), d=Vec3f0(1f0, 0f0, 0f0))

    intersects, interaction = Trace.intersect!(bvh2, ray1)
    @test intersects
    @test ray1.t_max ≈ 1f0
    @test ray1(ray1.t_max) ≈ Point3f0(-1f0, 0f0, 0f0)
    @test interaction.core.p ≈ Point3f0(-1f0, 0f0, 0f0)

    intersects, interaction = Trace.intersect!(bvh2, ray2)
    @test intersects
    @test ray2.t_max ≈ 17f0
    @test ray2(ray2.t_max) ≈ Point3f0(17f0, 18f0, 0f0)
    @test interaction.core.p ≈ Point3f0(-1f0, 0f0, 0f0)
end

@testset "LanczosSincFilter" begin
    l = Trace.LanczosSincFilter(Point2f0(4f0), 3f0)
    @test l(Point2f0(0f0)) ≈ 1f0
    @test l(Point2f0(4f0)) < 1f-6
    @test l(Point2f0(5f0)) ≈ 0f0
end
