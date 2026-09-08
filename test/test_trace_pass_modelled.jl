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
reason this file is about the trace alone: its indirect command is written by a
prepare kernel inside the recorded buffer, so a run re-executes it and rewrites
its own region.)

`Mantle.rebindable` knew, and refused: the per-run argument rewrite threw on a
plan with a `custom!` pass rather than quietly leaving it stale. Which meant
hardware ray tracing could not be recorded at all — the whole point of recording.

`custom!` has since been deleted from Mantle, along with `rebindable` and the
per-run rewrite itself. So the first of those assertions is now about a pass kind
that does not exist, and stays as a pass-kind whitelist: every pass of a hardware
render is one of the kinds the graph models. The second becomes what replaced the
rewrite — the trace is packed with the ADDRESS of `refs.sample_idx`, a
`Mantle.GPURef`, so the same recorded raygen renders a different sample every
run and the argument bytes never move.
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
        modelled = (:compute, :render, :copy, :update)
        for pl in plans
            for pp in pl.passes
                @test pp.pass.kind in modelled
            end
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

    @testset "plan $i: the trace's arguments live in the plan" for
            (i, pl) in enumerate(traced)
        t = first(d for pp in pl.passes for d in pp.dispatches
                  if d isa Mantle.CompiledTrace)
        @test t.argsize > 0
        plans = vp.state.plans

        Mantle.record!(pl)
        @test Mantle.recorded(pl)         # threw for a `custom!` plan

        bytes() = copy(unsafe_wrap(Array, pl.args.ptr + t.argoff, t.argsize))

        before = bytes()
        plans.perrun.sample_idx[] = Int32(3)
        Mantle.run!(pl)
        plans.perrun.sample_idx[] = Int32(99)
        Mantle.run!(pl)
        Mantle.waitfor!(pl)

        # The argument bytes do NOT move — that is the whole of what a `GPURef`
        # buys. Under `custom!` the trace packed its own arguments into queue
        # scratch the pool later rewound; under the per-run rewrite they were
        # host stores into memory a submission could still be reading. Now they
        # are written once, at `record!`, and the sample index reaches the raygen
        # through the address they hold.
        @test bytes() == before
    end

    close(vp)
end
