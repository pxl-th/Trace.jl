"""
The hardware ray-tracing pass is MODELLED by the graph, not smuggled past it.

Until 2026-08-30 `trace_pass!` declared its work with `Mantle.custom!` — the
escape hatch for a pass that "declares what it touches but not how". The trace
does not need it: it is one launch with arguments, which is exactly what
`Mantle.dispatch!` already models and what `Mantle.trace!` now models for a
shader binding table.

**What the escape hatch cost.** A `custom!` body packs its own arguments while it
runs. The old one called `get_arg_buffer(bq, n)`, which bump-allocates out of the
BATCH QUEUE's per-frame scratch, and pushed that address as a push constant — so
it was baked into the command buffer. The host writes those bytes; no GPU command
in the buffer does. A baked plan never runs the body again, so a replay reads a
scratch region whose bump pointer has since been rewound and handed to whatever
recorded next. Not stale values, aliased ones.

(An indirect DISPATCH is fine in the same situation, and the contrast is the
reason this file is about the trace alone: its indirect command is written by
`fast_prepare_indirect!`, a dispatch inside the captured buffer, so a replay
re-executes it and rewrites its own region.)

`Mantle.rebindable` knew, and refused: `rebind!` throws on a plan with a `custom!`
pass rather than quietly leaving it stale. Which meant hardware ray tracing could
not be baked at all — one of the two things `bake!` exists for.

So the assertions are: no plan of a hardware render has a `custom!` pass; every
one is rebindable; and `rebind!` on a BAKED plan really does rewrite the trace's
argument bytes, which is the part that would silently do nothing if the trace
ever went back to packing its own.
"""

using Test
using Hikari
using Lava, Mantle
using Raycore
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere, Point3f, Vec3f, Point2f

function _trace_scene(backend)
    scene = Hikari.Scene(; backend = backend, hw_accel = true)
    push!(scene, normal_mesh(Rect3f(Vec3f(-3, -3, -0.05), Vec3f(6, 6, 0.05))),
          Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.35), 0.35f0), 16)),
          Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.8f0, 0.3f0, 0.2f0)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    return scene
end

@testset "the hardware trace is a modelled pass" begin
    backend = Mantle.defaultbackend()
    scene = _trace_scene(backend)
    film = Hikari.Film(Point2f(48, 48))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5), Point3f(0, 0, 0.35),
                                      film; fov = 50f0)
    gpu_film = Hikari.Film(backend, film)
    vp = Hikari.VolPath(samples = 4, max_depth = 3, hw_accel = true)
    vp(scene, gpu_film, camera)

    img = Array(gpu_film.framebuffer)
    # First: the render is real. Every assertion below is about a plan, and a
    # plan that rendered nothing would satisfy all of them.
    @test 0.1 < count(px -> (px.r + px.g + px.b) > 1f-4, img) / length(img) < 0.99

    plans = Hikari.allplans(vp.state.plans)
    @test !isempty(plans)

    @testset "no plan reaches for the escape hatch" begin
        for pl in plans
            kinds = [pp.pass.kind for pp in pl.passes]
            @test (:custom in kinds) == false
            # The property that follows from it, asked directly: this is what
            # `bake!` needs and what threw before.
            @test Mantle.rebindable(pl)
        end
    end

    # How many there are is `chunking(max_depth)`'s business — a sample splits
    # into chunk and tail plans and each traces — so this asks for at least one
    # rather than pinning a number. Zero is the case that matters: it would mean
    # the hardware path was never taken and everything below passes by vacuum.
    traced = [pl for pl in plans
              if any(d -> d isa Mantle.CompiledTrace,
                     (d for pp in pl.passes for d in pp.dispatches))]
    @test !isempty(traced)

    @testset "plan $i: arguments live in the plan, and rebind! rewrites them" for
            (i, pl) in enumerate(traced)
        t = first(d for pp in pl.passes for d in pp.dispatches
                  if d isa Mantle.CompiledTrace)
        @test t.argsize > 0
        refs = vp.state.plans.refs

        Mantle.bake!(pl)
        @test pl.baked !== nothing        # threw for a `custom!` plan

        # `slotbase` after baking, not before: `bake!` takes the slot the
        # recording names for the rest of its life and stops rotating.
        off = Mantle.slotbase(pl.args) + t.argoff
        bytes() = copy(unsafe_wrap(Array, pl.args.ptr + off, t.argsize))

        refs.sample_idx[] = Int32(3)
        Mantle.rebind!(pl)
        a = bytes()

        refs.sample_idx[] = Int32(99)
        Mantle.rebind!(pl)
        b = bytes()

        # The sample index is one of the trace's arguments, so a rebind that
        # reached it changed the slot. Under `custom!` this was unreachable:
        # `rebind!` threw, and had it not, it would have walked a pass with
        # nothing in `dispatches` to repack.
        @test a != b
    end

    close(vp)
end
