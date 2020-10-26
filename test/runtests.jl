using Test
using GeometryBasics
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

@testset "Sphere bounds" begin
    core = Trace.ShapeCore(
        Trace.translate(Vec3f0(0)), Trace.translate(Vec3f0(0)), false,
    )
    s = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)

    sb = s |> Trace.object_bound
    @test sb[1] == Point3f0(-1f0)
    @test sb[2] == Point3f0(1f0)
end
