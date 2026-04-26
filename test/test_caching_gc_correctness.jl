# Hikari Caching, GC, and Correctness Tests
#
# Tests VolPath integrator caching (adapted scene, filter sampler, initial medium),
# VolPathState allocation/free lifecycle, WorkQueue GPU operations on Lava,
# close() cleanup, and rendering correctness across multiple renders.

using Test
using Hikari
using Lava
using KernelAbstractions
import KernelAbstractions as KA
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation
using LinearAlgebra

# Helper: create a minimal scene for testing
function _make_test_scene()
    scene = Hikari.Scene()

    # Floor
    floor_mat = Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.7f0, 0.7f0, 0.7f0))
    floor_mesh = normal_mesh(Rect3f(Vec3f(-2, -2, -0.01), Vec3f(4, 4, 0.01)))
    push!(scene, floor_mesh, floor_mat)

    # Sphere
    sphere_mat = Hikari.Diffuse(Kd=(0.8, 0.2, 0.2))
    sphere_mesh = normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.5), 0.5f0), 16))
    push!(scene, sphere_mesh, sphere_mat)

    # Light
    push!(scene, Hikari.PointLight(Point3f(0f0, 0f0, 3f0), Hikari.RGBSpectrum(20f0)))

    Hikari.sync!(scene)
    return scene
end

function _make_test_camera_film(; res=32)
    resolution = Point2f(res, res)
    film = Hikari.Film(resolution)
    camera = Hikari.PerspectiveCamera(
        Point3f(0f0, -3f0, 1f0), Point3f(0f0, 0f0, 0.5f0), film; fov=40f0,
    )
    return camera, film
end

@testset "Hikari Caching, GC & Correctness" begin

    # ── 1. VolPath close() frees all state ──
    @testset "VolPath close() lifecycle" begin
        @testset "close() clears all caches" begin
            vp = Hikari.VolPath(samples=1, max_depth=2)

            scene = _make_test_scene()
            camera, film = _make_test_camera_film()
            Hikari.clear!(film)

            # Render once to populate all caches
            vp(scene, film, camera)

            @test vp.state !== nothing
            # filter_sampler_gpu may be nothing for Gaussian filter with default sampler

            # Close should clear everything
            close(vp)

            @test vp.state === nothing
            @test vp.initial_medium_camera_pos === nothing
            @test vp.initial_medium_key === nothing
            @test vp.filter_sampler_gpu === nothing
        end

        @testset "close() is idempotent" begin
            vp = Hikari.VolPath(samples=1, max_depth=2)
            close(vp)  # No state yet
            close(vp)  # Should not error
            @test vp.state === nothing
        end
    end

    # ── 2. VolPathState allocation and free ──
    @testset "VolPathState allocation/free" begin
        backend = Lava.LavaBackend()

        @testset "state allocates all queues and buffers" begin
            GC.gc(true)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()
            baseline = length(Lava._live_buffers)

            scene = _make_test_scene()
            # Create a VolPathState directly
            state = Hikari.VolPathState(
                backend, 16, 16, scene.lights;
                max_depth=4, samples_per_pixel=1
            )
            Lava.vk_flush!(Lava.vk_context())

            # State should have allocated many GPU buffers
            after_alloc = length(Lava._live_buffers)
            @test after_alloc > baseline + 10  # At least 10+ buffers (queues + pixel buffers + tables)

            # Free state
            Hikari.free!(state)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()
            GC.gc(true)
            Lava.flush_deferred_frees!()

            after_free = length(Lava._live_buffers)
            # finalize() defers to GC which may not run immediately,
            # so we allow a small tolerance for pending frees
            @test after_free <= baseline + 10
        end

        @testset "double free! is safe" begin
            scene = _make_test_scene()
            state = Hikari.VolPathState(
                backend, 8, 8, scene.lights;
                max_depth=2, samples_per_pixel=1
            )
            Lava.vk_flush!(Lava.vk_context())
            Hikari.free!(state)
            # Second free should not crash (finalize on already-freed buffers is a no-op)
            Hikari.free!(state)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()
        end
    end

    # ── 3. WorkQueue on Lava backend ──
    @testset "WorkQueue on LavaBackend" begin
        backend = Lava.LavaBackend()

        @testset "push and read on GPU" begin
            queue = Hikari.WorkQueue{Int32}(backend, 256)

            @kernel function push_items!(queue)
                i = @index(Global)
                push!(queue, Int32(i * 10))
            end

            push_items!(backend)(queue; ndrange=8)
            Lava.vk_flush!(Lava.vk_context())

            @test length(queue) == 8
            items = sort(Array(queue.items)[1:8])
            @test items == Int32[10, 20, 30, 40, 50, 60, 70, 80]

            Hikari.free!(queue)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()
        end

        @testset "empty and reuse" begin
            queue = Hikari.WorkQueue{Int32}(backend, 64)

            @kernel function push_val!(queue, val)
                i = @index(Global)
                push!(queue, val)
            end

            push_val!(backend)(queue, Int32(42); ndrange=10)
            Lava.vk_flush!(Lava.vk_context())
            @test length(queue) == 10

            empty!(queue)
            @test length(queue) == 0

            push_val!(backend)(queue, Int32(99); ndrange=5)
            Lava.vk_flush!(Lava.vk_context())
            @test length(queue) == 5

            Hikari.free!(queue)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()
        end

        @testset "free! releases GPU memory" begin
            GC.gc(true)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()
            baseline = length(Lava._live_buffers)

            queue = Hikari.WorkQueue{Int32}(backend, 128)
            Lava.vk_flush!(Lava.vk_context())
            @test length(Lava._live_buffers) > baseline

            Hikari.free!(queue)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()
            @test length(Lava._live_buffers) == baseline
        end
    end

    # ── 4. Filter sampler caching ──
    @testset "filter sampler cache" begin
        @testset "filter sampler cached on struct" begin
            # LanczosSinc filter produces GPUFilterSamplerData that gets cached
            vp = Hikari.VolPath(samples=1, max_depth=2,
                                filter=Hikari.LanczosSincFilter(Point2f(1.0f0), 3.0f0))

            @test vp.filter_sampler_gpu === nothing

            scene = _make_test_scene()
            camera, film = _make_test_camera_film()
            Hikari.clear!(film)
            vp(scene, film, camera)

            # After render, should be cached
            # Filter sampler data should be cached from film after render
            @test vp.filter_sampler_gpu !== nothing

            close(vp)
        end
    end

    # ── 6. State reuse across samples ──
    @testset "state reuse across samples" begin
        @testset "state persists between render! calls" begin
            vp = Hikari.VolPath(samples=2, max_depth=2)
            scene = _make_test_scene()
            camera, film = _make_test_camera_film()
            Hikari.clear!(film)

            # First render populates state
            vp(scene, film, camera)
            state_after_first = vp.state
            @test state_after_first !== nothing

            # Clear and render again — state should be reused (same dimensions)
            Hikari.clear!(film)
            vp(scene, film, camera)
            @test vp.state === state_after_first

            close(vp)
        end
    end

    # ── 7. Rendering correctness ──
    @testset "rendering correctness" begin
        @testset "output has valid pixel values" begin
            vp = Hikari.VolPath(samples=4, max_depth=4)
            scene = _make_test_scene()
            camera, film = _make_test_camera_film(; res=32)
            Hikari.clear!(film)

            vp(scene, film, camera)

            img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)
            img_arr = Array(img)

            @test size(img_arr) == (32, 32)

            # No NaN or Inf
            @test !any(px -> isnan(px.r) || isnan(px.g) || isnan(px.b), img_arr)
            @test !any(px -> isinf(px.r) || isinf(px.g) || isinf(px.b), img_arr)

            # Something was rendered (not all black)
            mean_val = sum(px -> (px.r + px.g + px.b) / 3f0, img_arr) / length(img_arr)
            @test mean_val > 0.001f0

            close(vp)
        end

        @testset "deterministic across runs with same seed" begin
            scene = _make_test_scene()

            vp1 = Hikari.VolPath(samples=2, max_depth=4)
            camera1, film1 = _make_test_camera_film(; res=16)
            Hikari.clear!(film1)
            vp1(scene, film1, camera1)
            img1 = Array(Hikari.postprocess!(film1; exposure=1.0f0, tonemap=nothing, gamma=1.0f0))

            close(vp1)

            vp2 = Hikari.VolPath(samples=2, max_depth=4)
            camera2, film2 = _make_test_camera_film(; res=16)
            Hikari.clear!(film2)
            vp2(scene, film2, camera2)
            img2 = Array(Hikari.postprocess!(film2; exposure=1.0f0, tonemap=nothing, gamma=1.0f0))

            close(vp2)

            # Same scene + camera + spp should produce identical results
            # (SobolRNG uses deterministic seed 0 by default)
            @test img1 == img2
        end

        @testset "more samples produces non-zero output" begin
            scene = _make_test_scene()

            vp = Hikari.VolPath(samples=4, max_depth=4)
            camera, film = _make_test_camera_film(; res=16)
            Hikari.clear!(film)
            vp(scene, film, camera)
            img = Array(Hikari.postprocess!(film; exposure=1.0f0, tonemap=nothing, gamma=1.0f0))
            close(vp)

            # 4 samples should produce a reasonable image
            mean_val = sum(px -> Float64(px.r + px.g + px.b) / 3.0, img) / length(img)
            @test mean_val > 0.001
        end
    end

    # ── 8. Memory stability across repeated renders ──
    @testset "memory stability" begin
        @testset "buffer count stable across renders" begin
            GC.gc(true)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()

            scene = _make_test_scene()
            vp = Hikari.VolPath(samples=1, max_depth=2)
            camera, film = _make_test_camera_film()

            # Warm-up render
            Hikari.clear!(film)
            vp(scene, film, camera)
            Lava.vk_flush!(Lava.vk_context())
            GC.gc(true)
            Lava.flush_deferred_frees!()
            baseline = length(Lava._live_buffers)

            # Multiple renders — buffer count should not grow
            for _ in 1:5
                Hikari.clear!(film)
                vp(scene, film, camera)
                Lava.vk_flush!(Lava.vk_context())
            end
            GC.gc(true)
            Lava.flush_deferred_frees!()
            after = length(Lava._live_buffers)

            @test after == baseline

            close(vp)
            Lava.vk_flush!(Lava.vk_context())
            Lava.flush_deferred_frees!()
        end
    end

    # ── 9. LavaArray resize! does not leak ──
    @testset "resize! does not leak buffers" begin
        GC.gc(true)
        Lava.vk_flush!(Lava.vk_context())
        Lava.flush_deferred_frees!()
        baseline = length(Lava._live_buffers)

        # resize! replaces the internal DataRef — old buffer must be freed
        a = Lava.LavaArray{Int32}(undef, 10)
        Lava.vk_flush!(Lava.vk_context())
        after_alloc = length(Lava._live_buffers)
        @test after_alloc == baseline + 1

        resize!(a, 100)
        Lava.vk_flush!(Lava.vk_context())
        Lava.flush_deferred_frees!()
        after_resize = length(Lava._live_buffers)
        # Should still be baseline + 1 (new buffer), not baseline + 2 (old leaked)
        @test after_resize == baseline + 1

        # Multiple resizes should not accumulate
        for sz in [200, 50, 500, 10]
            resize!(a, sz)
        end
        Lava.vk_flush!(Lava.vk_context())
        Lava.flush_deferred_frees!()
        after_multi = length(Lava._live_buffers)
        @test after_multi == baseline + 1

        # Free the array itself
        finalize(a)
        Lava.vk_flush!(Lava.vk_context())
        Lava.flush_deferred_frees!()
        after_free = length(Lava._live_buffers)
        @test after_free == baseline
    end

    # ── 10. Film lifecycle ──
    @testset "Film lifecycle" begin
        @testset "film clear resets iteration index" begin
            film = Hikari.Film(Point2f(16, 16))
            film.iteration_index[] = Int32(42)
            Hikari.clear!(film)
            @test film.iteration_index[] == Int32(0)
        end

        @testset "film free releases memory" begin
            film = Hikari.Film(Point2f(16, 16))
            Hikari.free!(film)
            # Should not crash on double free
            Hikari.free!(film)
        end
    end
end
