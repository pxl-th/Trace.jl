using Test
using GeometryBasics
using LinearAlgebra
using Trace
using FileIO
using ImageCore

function check_scene_average(scene_file::String, target::Float32)
    scene = scene_file |> load |> channelview
    average = sum(scene) / length(scene)
    @test average ≈ target
end

@testset "Test Bounds2 iteration" begin
    b = Trace.Bounds2(Point2f0(1f0, 3f0), Point2f0(4f0, 4f0))
    targets = [
        Point2f0(1f0, 3f0), Point2f0(2f0, 3f0), Point2f0(3f0, 3f0), Point2f0(4f0, 3f0),
        Point2f0(1f0, 4f0), Point2f0(2f0, 4f0), Point2f0(3f0, 4f0), Point2f0(4f0, 4f0),
    ]
    @test length(b) == 8
    for (p, t) in zip(b, targets)
        @test p == t
    end

    b = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
    targets = [
        Point2f0(-1f0, -1f0), Point2f0(0f0, -1f0), Point2f0(1f0, -1f0),
        Point2f0(-1f0, 0f0), Point2f0(0f0, 0f0), Point2f0(1f0, 0f0),
        Point2f0(-1f0, 1f0), Point2f0(0f0, 1f0), Point2f0(1f0, 1f0),
    ]
    @test length(b) == 9
    for (p, t) in zip(b, targets)
        @test p == t
    end
end

@testset "Ray-Bounds intersection" begin
    b = Trace.Bounds3(Point3f0(1), Point3f0(2))
    b_neg = Trace.Bounds3(Point3f0(-2), Point3f0(-1))
    r0 = Trace.Ray(o=Point3f0(0), d=Vec3f0(1, 0, 0))
    r1 = Trace.Ray(o=Point3f0(0), d=Vec3f0(1))
    ri = Trace.Ray(o=Point3f0(1.5), d=Vec3f0(1, 1, 0))

    r, t0, t1 = Trace.intersect(b, r1)
    @test r && t0 ≈ 1f0 && t1 ≈ 2f0
    r, t0, t1 = Trace.intersect(b, r0)
    @test !r && t0 ≈ 0f0 && t1 ≈ 0f0
    r, t0, t1 = Trace.intersect(b, ri)
    @test r && t0 ≈ 0f0 && t1 ≈ 0.5f0

    # Test intersection with precomputed direction reciprocal.
    inv_dir = 1f0 ./ r1.d
    dir_is_negative = r1.d |> Trace.is_dir_negative
    @test Trace.intersect_p(b, r1, inv_dir, dir_is_negative)
    @test !Trace.intersect_p(b_neg, r1, inv_dir, dir_is_negative)
end

@testset "Sphere bound" begin
    core = Trace.ShapeCore(Trace.translate(Vec3f0(0)), false)
    s = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)

    sb = s |> Trace.object_bound
    @test sb[1] == Point3f0(-1f0)
    @test sb[2] == Point3f0(1f0)
end

@testset "Ray-Sphere insersection" begin
    # Sphere at the origin.
    core = Trace.ShapeCore(Trace.Transformation(), false)
    s = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)
    r0 = Trace.Ray(o=Point3f0(0f0), d=Vec3f0(0f0, 1f0, 0f0))
    r = Trace.Ray(o=Point3f0(0, -2, 0), d=Vec3f0(0, 1, 0))

    i, t, interaction = Trace.intersect(s, r, false)
    ip = Trace.intersect_p(s, r, false)

    @test i == ip
    @test t ≈ 1f0
    @test r(t) ≈ Point3f0(0, -1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f0(0, -1, 0) # Object intersection.
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0
    # Test ray inside a sphere.
    i, t, interaction = Trace.intersect(s, r0, false)
    @test i
    @test t ≈ 1f0
    @test r0(t) ≈ Point3f0(0f0, 1f0, 0f0)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0

    # Translated sphere.
    core = Trace.ShapeCore(Trace.translate(Vec3f0(0, 2, 0)), false)
    s = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)
    r = Trace.Ray(o=Point3f0(0, 0, 0), d=Vec3f0(0, 1, 0))

    i, t, interaction = Trace.intersect(s, r, false)
    ip = Trace.intersect_p(s, r, false)

    @test i == ip
    @test t ≈ 1f0
    @test r(t) ≈ Point3f0(0, 1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f0(0, 1, 0) # Object intesection.
end

@testset "Test triangle" begin
    core = Trace.ShapeCore(Trace.translate(Vec3f0(0, 0, 2)), false)
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
        core = Trace.ShapeCore(Trace.translate(Vec3f0(i, i, 0)), false)
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
    @test interaction.core.p ≈ Point3f0(17f0, 18f0, 0f0)
end

@testset "LanczosSincFilter" begin
    l = Trace.LanczosSincFilter(Point2f0(4f0), 3f0)
    @test l(Point2f0(0f0)) ≈ 1f0
    @test l(Point2f0(4f0)) < 1f-6
    @test l(Point2f0(5f0)) ≈ 0f0
end

@testset "Film testing" begin
    filter = Trace.LanczosSincFilter(Point2f0(4f0), 3f0)
    film = Trace.Film(
        Point2f0(1920f0, 1080f0), Trace.Bounds2(Point2f0(0f0), Point2f0(1f0)),
        filter, 35f0, 1f0, "output.png",
    )
    @test size(film.pixels) == (1080, 1920)
    @test Trace.get_sample_bounds(film) == Trace.Bounds2(Point2f0(-3f0), Point2f0(1924f0, 1084f0))
end

@testset "FilmTile testing" begin
    filter = Trace.LanczosSincFilter(Point2f0(4f0), 3f0)
    film = Trace.Film(
        Point2f0(1920f0, 1080f0), Trace.Bounds2(Point2f0(0f0), Point2f0(1f0)),
        filter, 35f0, 1f0, "output.png",
    )
    # Test tile from start of the film.
    tile = Trace.FilmTile(film, Trace.Bounds2(Point2f0(1f0), Point2f0(10f0)))
    @test size(tile.pixels) == (14, 14)
    @test tile.bounds == Trace.Bounds2(Point2f0(1f0), Point2f0(14f0))
    # Pixels in 1:5 radius should be affected.
    for i in 1:5
        @test tile.pixels[i, i].filter_weight_sum ≈ 0f0
    end
    Trace.add_sample!(tile, Point2f0(1f0), Trace.RGBSpectrum(1f0))
    for (i, j) in zip(1:4, 2:5)
        @test tile.pixels[i, i].filter_weight_sum > 0 && tile.pixels[j, j].filter_weight_sum > 0
        @test tile.pixels[i, i].filter_weight_sum > tile.pixels[j, j].filter_weight_sum
    end
    # Merging tile back to film.
    for i in 1:5
        @test film.pixels[i, i].filter_weight_sum ≈ 0f0
    end
    Trace.merge_film_tile!(film, tile)
    for (i, j) in zip(1:4, 2:5)
        @test film.pixels[i, i].filter_weight_sum > 0 && film.pixels[j, j].filter_weight_sum > 0
        @test film.pixels[i, i].filter_weight_sum > film.pixels[j, j].filter_weight_sum
    end

    # Test shifted tile.
    tile = Trace.FilmTile(film, Trace.Bounds2(Point2f0(10f0), Point2f0(60f0)))
    @test size(tile.pixels) == (59, 59) # Account for filter radius shift.
    @test tile.bounds == Trace.Bounds2(Point2f0(6f0), Point2f0(64f0))
    # Add to [20, 20] pixel
    #   -> [16:24] radius of contribution on the film
    #   -> [16 - 6 + 1:24 - 6 + 1] = [11:19] radius of contribution on the film tile.
    for i in 11:19
        @test tile.pixels[i, i].filter_weight_sum ≈ 0f0
    end
    Trace.add_sample!(tile, Point2f0(20f0), Trace.RGBSpectrum(1f0))
    # Symmetrical.
    for (i, j) in zip(11:14, 18:-1:15)
        @test tile.pixels[i, i].filter_weight_sum ≈ tile.pixels[j, j].filter_weight_sum
    end
    # Increasing from left-to-right.
    for (i, j) in zip(11:13, 12:14)
        @test tile.pixels[i, i].filter_weight_sum > 0 && tile.pixels[j, j].filter_weight_sum > 0
        @test tile.pixels[i, i].filter_weight_sum < tile.pixels[j, j].filter_weight_sum
    end
    # Decreasing from right-to-left.
    for (i, j) in zip(16:18, 17:19)
        @test tile.pixels[i, i].filter_weight_sum > 0 && tile.pixels[j, j].filter_weight_sum > 0
        @test tile.pixels[i, i].filter_weight_sum > tile.pixels[j, j].filter_weight_sum
    end
    # Merging tile back to film.
    for i in 16:24
        @test film.pixels[i, i].filter_weight_sum ≈ 0f0
    end
    Trace.merge_film_tile!(film, tile)
    # Symmetrical.
    for (i, j) in zip(16:19, 23:-1:20)
        @test film.pixels[i, i].filter_weight_sum ≈ film.pixels[j, j].filter_weight_sum
    end
    # Increasing from left-to-right.
    for (i, j) in zip(16:18, 17:19)
        @test film.pixels[i, i].filter_weight_sum > 0 && film.pixels[j, j].filter_weight_sum > 0
        @test film.pixels[i, i].filter_weight_sum < film.pixels[j, j].filter_weight_sum
    end
    # Decreasing from right-to-left.
    for (i, j) in zip(20:23, 21:24)
        @test film.pixels[i, i].filter_weight_sum > 0 && film.pixels[j, j].filter_weight_sum > 0
        @test film.pixels[i, i].filter_weight_sum > film.pixels[j, j].filter_weight_sum
    end
end

@testset "Frensel Dielectric" begin
    # Vacuum gives no reflectance.
    @test Trace.frensel_dielectric(1f0, 1f0, 1f0) ≈ 0f0
    @test Trace.frensel_dielectric(0.5f0, 1f0, 1f0) ≈ 0f0
    # Vacuum-diamond -> total reflection.
    @test Trace.frensel_dielectric(cos(π / 4f0), 1f0, 2.42f0) ≈ 1f0
end

@testset "Frensel Conductor" begin
    s = Trace.RGBSpectrum(1f0)
    @test Trace.frensel_conductor(0f0, s, s, s) == s
    @test all(Trace.frensel_conductor(cos(π / 4f0), s, s, s).c .> 0f0)
    @test all(Trace.frensel_conductor(1f0, s, s, s).c .> 0f0)
end

@testset "SpecularReflection" begin
    sr = Trace.SpecularReflection(Trace.RGBSpectrum(1f0), Trace.FrenselNoOp())
    @test sr & Trace.BSDF_REFLECTION
    @test sr & Trace.BSDF_SPECULAR
    @test sr & (Trace.BSDF_SPECULAR | Trace.BSDF_REFLECTION)
end

@testset "SpecularTransmission" begin
    st = Trace.SpecularTransmission{Trace.RGBSpectrum, Trace.Radiance}(
        Trace.RGBSpectrum(1f0), 1f0, 1f0,
        Trace.FrenselDielectric(1f0, 1f0),
    )
    @test st & Trace.BSDF_SPECULAR
    @test st & Trace.BSDF_TRANSMISSION
    @test st & (Trace.BSDF_SPECULAR | Trace.BSDF_TRANSMISSION)
end

@testset "Perspective Camera" begin
    filter = Trace.LanczosSincFilter(Point2f0(4f0), 3f0)
    film = Trace.Film(
        Point2f0(1920f0, 1080f0), Trace.Bounds2(Point2f0(0f0), Point2f0(1f0)),
        filter, 35f0, 1f0, "output.png",
    )
    camera = Trace.PerspectiveCamera(
        Trace.translate(Vec3f0(0)), Trace.Bounds2(Point2f0(0), Point2f0(10)),
        0f0, 1f0, 0f0, 700f0, 45f0, film,
    )

    sample1 = Trace.CameraSample(Point2f0(1f0), Point2f0(1f0), 0f0)
    ray1, contribution = Trace.generate_ray(camera, sample1)
    sample2 = Trace.CameraSample(
        Point2f0(film.resolution[1]), Point2f0(film.resolution[2]), 0f0,
    )
    ray2, contribution = Trace.generate_ray(camera, sample2)

    @test contribution == 1f0
    @test ray1.o == ray2.o == Point3f0(0f0)
    @test ray1.time == ray2.time == camera.core.core.shutter_open
    @test ray1.d[1] < ray2.d[1] && ray1.d[2] < ray2.d[2]
    @test argmax(abs.(ray1.d)) == argmax(abs.(ray2.d)) == 3

    ray_differential, contribution = Trace.generate_ray_differential(
        camera, sample1,
    )
    @test ray_differential.has_differentials
    @test ray_differential.o == Point3f0(0f0)
    @test ray_differential.d ≈ Point3f0(ray1.d)

    @test ray_differential.rx_direction[1] > ray_differential.d[1]
    @test ray_differential.rx_direction[2] ≈ ray_differential.d[2]
    @test ray_differential.ry_direction[1] ≈ ray_differential.d[1]
    @test ray_differential.ry_direction[2] > ray_differential.d[2]
end

# @testset "Analytic scene" begin
#     # Unit sphere, Kd = 0.5, point light I = π at center
#     # With GI, should have radiance of 1.
#     material = Trace.MatteMaterial(
#         Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
#         Trace.ConstantTexture(0f0),
#     )
#     core = Trace.ShapeCore(Trace.Transformation(), true)
#     sphere = Trace.Sphere(core, 1f0, -1f0, 1f0, 360f0)
#     primitive = Trace.GeometricPrimitive(sphere, material)
#     bvh = Trace.BVHAccel{Trace.SAH}([primitive])

#     lights = [Trace.PointLight(
#         Trace.Transformation(), Trace.RGBSpectrum(Float32(π)),
#     )]
#     scene = Trace.Scene(lights, bvh)
#     # Construct Film and Camera.
#     resolution = Point2f0(10f0, 10f0)
#     filter = Trace.LanczosSincFilter(Point2f0(4f0), 3f0)
#     scene_file = "test-output.png"
#     film = Trace.Film(
#         resolution, Trace.Bounds2(Point2f0(0f0), Point2f0(1f0)),
#         filter, 1f0, 1f0, scene_file,
#     )
#     screen = Trace.Bounds2(Point2f0(-1f0), Point2f0(1f0))
#     camera = Trace.PerspectiveCamera(
#         Trace.Transformation(), screen, 0f0, 1f0, 0f0, 10f0, 45f0, film,
#     )

#     sampler = Trace.UniformSampler(1)
#     integrator = Trace.WhittedIntegrator(camera, sampler, 1)
#     scene |> integrator

#     check_scene_average(scene_file, 1f0)
# end
