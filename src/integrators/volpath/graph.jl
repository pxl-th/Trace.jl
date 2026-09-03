# One volpath sample, as Mantle graphs.
#
# The wavefront loop is ~20 dispatches a round whose only ordering constraints
# are which queue each stage fills and which one it drains. Those constraints
# used to be expressed by the order of the calls plus two hand-placed
# `concurrent_dispatch_group` / `concurrent_indirect_group` scopes asserting
# which dispatches were independent — an assertion nothing checked, and the one
# that was wrong (a prepare-indirect read racing its own producer) cost ~15 % of
# the energy on a scene where the queue happened to empty mid-pipeline.
#
# Here each stage declares what it reads and what it writes, and the barriers
# between them are derived from that. A stage that shares nothing with its
# neighbour gets no barrier and the two overlap; a stage that drains what
# another filled waits for exactly those bytes.
#
# Three graphs, because two of the boundaries are the host's:
#
#   setup       clear the film, reset the ray queue, generate camera rays
#   round       one bounce, built twice for the two directions of the ray
#               queue ping-pong
#   accumulate  this sample's spectral radiance into the RGB accumulators
#
# and `finalize` for the divide, which is its own plan because a batched caller
# runs it once after every sample rather than per sample.
#
# The bounce loop stays a host loop: a `Plan` is compiled once and run many
# times, so a plan per round expresses it, and the early exit reads a device
# counter and breaks — which no graph-level loop node would make cheaper.

# What is NOT here: the queues are Hikari's own allocations, read by the passes
# as foreign buffers. A graph can order accesses to a buffer it does not own; it
# cannot adopt one into its arena, and aliasing is the arena's. Moving them was
# the obvious last step of the port and it is measured to be worth nothing on
# this loop — `benchmarks/queue_aliasing/aliasing.jl` puts every intermediate
# queue in the placer's hands at 1280x1080 and gets 174 MiB back for ONE round
# and exactly zero for two or more, because every queue is refilled every round
# and so is live from the first to the last. The shipped chunk is eight.

import Mantle
import StructArrays

# ─────────────────────────────────────────────────────────────────────────────
# What a stage touches
# ─────────────────────────────────────────────────────────────────────────────

"""
    devicebuffers!(acc, x) -> acc

Every device buffer reachable from `x`, appended to `acc`.

A barrier is scoped to a buffer, so a stage declares the leaves rather than the
container that holds them: a work queue is a payload plus an atomic counter, and
on the SOA path that payload is one array per field of the work item, several
levels deep.
"""
devicebuffers!(acc, x::AbstractArray) = (push!(acc, x); acc)
devicebuffers!(acc, x::StructArrays.StructArray) =
    (foreach(c -> devicebuffers!(acc, c), StructArrays.components(x)); acc)
devicebuffers!(acc, q::WorkQueue) = devicebuffers!(devicebuffers!(acc, q.items), q.size)
devicebuffers!(acc, m::MultiTypeWorkQueue) =
    (foreach(q -> devicebuffers!(acc, q), m.queues); acc)

"""
    use!(pass, x; read, write)

Declare `x` — a buffer, a work queue, or a whole multi-type queue — at the
access this stage uses it with. Returns nothing: what the kernel receives is the
container, and what the graph orders on are its buffers.
"""
function use!(p, x; read::Bool = false, write::Bool = false, unordered::Bool = false)
    for b in devicebuffers!(Any[], x)
        Mantle.use(p, b; read = read, write = write, unordered = unordered)
    end
    return nothing
end

"""
    accumulates!(pass, buffer)

Declare a buffer this stage only ever adds into, atomically.

Six stages of a round do exactly one thing to the per-pixel radiance —
`atomic +=` — and the order they do it in does not change the sum. Declared as
an ordinary read-write that is a hazard between every pair of them, so the
escaped, emitter and shading stages are ordered against each other even though
their queues are disjoint and nothing else connects them. `Unordered` is the
vocabulary for exactly this, and it is checked on both sides: only two stages
that BOTH declare it may overlap, so the film clear before them and the
accumulate after them still get their barriers.

Float addition is not associative, so the sum's last bits depend on the order
the additions land in — which is already true between invocations of one
dispatch and is why the pbrt gate is a tolerance and not an equality.
"""
accumulates!(p, x) = use!(p, x; read = true, write = true, unordered = true)

# ─────────────────────────────────────────────────────────────────────────────
# The values a recorded plan reads per render
# ─────────────────────────────────────────────────────────────────────────────

"""
    render_refs(...) -> NamedTuple of Ref

What changes between two calls to `render!` while the plans stay the same: the
adapted scene (rebuilt every call), the sample index, and everything the caller
can set on the integrator between samples.

They are `Ref`s because a compiled dispatch resolves its arguments once and a
recording reads them per run — `Mantle` dereferences a `Ref` at record time for
exactly this. The types have to stay put, which is what [`PlanKey`](@ref)
checks: a scene whose adapted accel is a different type gets its own plans
rather than arguments packed to the wrong layout.
"""
render_refs(accel, media_interfaces, media, materials, lights,
            camera, camera_needs_time::Bool, camera_needs_lens::Bool,
            initial_medium, filter_params, filter_sampler,
            regularize::Bool, samples_per_pixel::Int32,
            max_component_value::Float32) =
    (accel = Ref(accel), media_interfaces = Ref(media_interfaces),
     media = Ref(media), materials = Ref(materials), lights = Ref(lights),
     camera = Ref(camera), camera_needs_time = Ref(camera_needs_time),
     camera_needs_lens = Ref(camera_needs_lens),
     initial_medium = Ref(initial_medium),
     filter_params = Ref(filter_params), filter_sampler = Ref(filter_sampler),
     sample_idx = Ref(Int32(0)), regularize = Ref(regularize),
     samples_per_pixel = Ref(samples_per_pixel),
     max_component_value = Ref(max_component_value))

# ─────────────────────────────────────────────────────────────────────────────
# Kernels the graph needs and the KA path did not
# ─────────────────────────────────────────────────────────────────────────────

# `KA.fill!` is a launch rather than a `@kernel`, and a pass records kernels.
# Writing the fill as one is also what lets it share a pass with the counter
# reset: the two touch disjoint memory, which is the fact the hand-placed
# `concurrent_dispatch_group` around them used to assert.
@kernel inbounds = true function vp_clear_kernel!(dst, value)
    i = @index(Global)
    dst[i] = value
end

# ─────────────────────────────────────────────────────────────────────────────
# The passes
# ─────────────────────────────────────────────────────────────────────────────

"""Zero every counter the coming round fills. One dispatch for all of them —
a `fill!` per queue was ~20 commands and their barriers per round."""
function reset_pass!(g, state::VolPathState, nxt::WorkQueue)
    queues = round_queues(state, nxt)
    counters = _collect_size_counters((), queues...)
    Mantle.compute!(g, "reset") do p
        for c in counters
            Mantle.use(p, c; write = true)
        end
        Mantle.dispatch!(p, zero_size_counters_kernel!, (counters,), 1; group = 1)
    end
end

"""Every queue a round fills, so the reset covers all of them and nothing
carries a count over from the previous bounce."""
function round_queues(state::VolPathState, nxt::WorkQueue)
    pmq = state.per_material_queue
    base = (nxt, state.medium_sample_queue, state.medium_scatter_queue,
            state.hit_surface_queue, state.shadow_queue, state.escaped_queue,
            state.hit_area_light_queue)
    return pmq === nothing ? base : (base..., pmq)
end

"""
Trace the live rays and shade whatever needs no queue of its own.

Two implementations on the accel axis, and the pass kind differs with them: the
software BVH and inline ray queries are one compute kernel over the ray queue,
while the hardware RT pipeline is a `vkCmdTraceRaysIndirect` with an SBT, which
is not a dispatch. `custom!` is the pass for that — the body records through the
backend and the graph still orders it, because what it touches is declared here
either way.
"""
function trace_pass!(g, accel, state::VolPathState, refs, cur::WorkQueue, nxt::WorkQueue)
    Mantle.compute!(g, "trace") do p
        trace_uses!(p, state, cur, nxt)
        Mantle.dispatch!(p, workqueue_map_kernel!,
                         (vp_trace_and_shade_kernel!, cur, nxt,
                          state.escaped_queue, state.medium_sample_queue,
                          state.per_material_queue, state.hit_surface_queue,
                          state.hit_area_light_queue, state.pixel_L,
                          refs.accel, refs.media_interfaces, refs.media,
                          refs.materials, refs.lights,
                          state.rgb2spec_table,
                          state.bvh_nodes, state.infinite_light_indices,
                          state.light_to_bit_trail,
                          state.num_infinite_lights, state.num_bvh_lights,
                          state.num_lights,
                          state.max_depth, refs.regularize,
                          state.sobol_rng, refs.sample_idx,
                          refs.camera, refs.samples_per_pixel, state.rr_depth),
                         Mantle.DeviceRange(cur.size; max = Int(cur.capacity));
                         group = DEFAULT_WORKGROUPSIZE)
    end
end

function trace_uses!(p, state::VolPathState, cur::WorkQueue, nxt::WorkQueue)
    use!(p, cur; read = true)
    use!(p, nxt; read = true, write = true)
    use!(p, state.escaped_queue; read = true, write = true)
    use!(p, state.medium_sample_queue; read = true, write = true)
    use!(p, state.per_material_queue; read = true, write = true)
    use!(p, state.hit_surface_queue; read = true, write = true)
    use!(p, state.hit_area_light_queue; read = true, write = true)
    accumulates!(p, state.pixel_L)
    return nothing
end

"""Delta tracking through the media the trace deferred. Fills the scatter queue
and, for rays that leave their medium and land on a surface, the same surface
queues the trace fills."""
function medium_sample_pass!(g, state::VolPathState, refs, nxt::WorkQueue)
    Mantle.compute!(g, "medium-sample") do p
        use!(p, state.medium_sample_queue; read = true)
        use!(p, state.medium_scatter_queue; read = true, write = true)
        use!(p, state.per_material_queue; read = true, write = true)
        use!(p, state.hit_surface_queue; read = true, write = true)
        use!(p, state.hit_area_light_queue; read = true, write = true)
        use!(p, nxt; read = true, write = true)
        use!(p, state.escaped_queue; read = true, write = true)
        accumulates!(p, state.pixel_L)
        Mantle.dispatch!(p, workqueue_map_kernel!,
                         (vp_sample_medium_kernel!, state.medium_sample_queue,
                          state.medium_scatter_queue, state.per_material_queue,
                          state.hit_surface_queue, state.hit_area_light_queue,
                          nxt, state.escaped_queue, state.pixel_L,
                          refs.media, refs.materials, state.rgb2spec_table,
                          state.max_depth),
                         Mantle.DeviceRange(state.medium_sample_queue.size;
                                            max = Int(state.medium_sample_queue.capacity));
                         group = DEFAULT_WORKGROUPSIZE)
    end
end

"""Direct lighting at a real scatter event. The only producer of shadow rays."""
function medium_dl_pass!(g, state::VolPathState, refs)
    Mantle.compute!(g, "medium-dl") do p
        use!(p, state.medium_scatter_queue; read = true)
        use!(p, state.shadow_queue; read = true, write = true)
        Mantle.dispatch!(p, workqueue_map_kernel!,
                         (vp_medium_direct_lighting_kernel!,
                          state.medium_scatter_queue, state.shadow_queue,
                          refs.lights, state.rgb2spec_table,
                          state.bvh_nodes, state.infinite_light_indices,
                          state.num_infinite_lights, state.num_bvh_lights,
                          state.num_lights,
                          state.sobol_rng, refs.sample_idx),
                         Mantle.DeviceRange(state.medium_scatter_queue.size;
                                            max = Int(state.medium_scatter_queue.capacity));
                         group = DEFAULT_WORKGROUPSIZE)
    end
end

"""The phase-function bounce out of a scatter event."""
function medium_scatter_pass!(g, state::VolPathState, refs, nxt::WorkQueue)
    Mantle.compute!(g, "medium-scatter") do p
        use!(p, state.medium_scatter_queue; read = true)
        use!(p, nxt; read = true, write = true)
        Mantle.dispatch!(p, workqueue_map_kernel!,
                         (vp_medium_scatter_kernel!, state.medium_scatter_queue,
                          nxt, state.max_depth, state.sobol_rng, refs.sample_idx),
                         Mantle.DeviceRange(state.medium_scatter_queue.size;
                                            max = Int(state.medium_scatter_queue.capacity));
                         group = DEFAULT_WORKGROUPSIZE)
    end
end

"""Rays that hit nothing: the environment lights, with MIS."""
function escaped_pass!(g, state::VolPathState, refs)
    Mantle.compute!(g, "escaped") do p
        use!(p, state.escaped_queue; read = true)
        accumulates!(p, state.pixel_L)
        Mantle.dispatch!(p, workqueue_map_kernel!,
                         (vp_handle_escaped_rays_kernel!, state.escaped_queue,
                          state.pixel_L, state.rgb2spec_table, refs.lights,
                          state.bvh_nodes, state.light_to_bit_trail,
                          state.infinite_light_indices,
                          state.num_infinite_lights, state.num_bvh_lights),
                         Mantle.DeviceRange(state.escaped_queue.size;
                                            max = Int(state.escaped_queue.capacity));
                         group = DEFAULT_WORKGROUPSIZE)
    end
end

"""Emission MIS for indirect rays that landed on an area light — pbrt-v4's
"Handle emitters hit by indirect rays"."""
function emitters_pass!(g, state::VolPathState, refs)
    Mantle.compute!(g, "emitters") do p
        use!(p, state.hit_area_light_queue; read = true)
        accumulates!(p, state.pixel_L)
        Mantle.dispatch!(p, workqueue_map_kernel!,
                         (vp_handle_emitters_kernel!, state.hit_area_light_queue,
                          state.pixel_L, refs.lights, state.rgb2spec_table,
                          state.bvh_nodes, state.light_to_bit_trail,
                          state.num_infinite_lights, state.num_bvh_lights,
                          state.num_lights),
                         Mantle.DeviceRange(state.hit_area_light_queue.size;
                                            max = Int(state.hit_area_light_queue.capacity));
                         group = DEFAULT_WORKGROUPSIZE)
    end
end

"""
Surface shading, one dispatch per concrete material type.

They go in ONE pass because they are mutually independent — each drains its own
typed queue and writes atomically-claimed slots in the shared next-ray queue —
and a pass is the unit Mantle emits barriers between. That is what the
hand-placed `concurrent_indirect_group` used to say; here it follows from the
dispatches sharing a pass, and the backend fuses their prepares into one.
"""
function shade_pass!(g, state::VolPathState, refs, nxt::WorkQueue)
    Mantle.compute!(g, "shade") do p
        use!(p, state.per_material_queue; read = true)
        use!(p, state.hit_surface_queue; read = true)
        use!(p, nxt; read = true, write = true)
        accumulates!(p, state.pixel_L)
        for q in state.per_material_queue.queues
            Mantle.dispatch!(p, workqueue_map_kernel!,
                             (vp_shade_material_kernel!, q,
                              state.hit_surface_queue, nxt, state.pixel_L,
                              refs.accel, refs.media_interfaces, refs.media,
                              refs.materials, refs.lights, state.rgb2spec_table,
                              state.bvh_nodes, state.infinite_light_indices,
                              state.light_to_bit_trail,
                              state.num_infinite_lights, state.num_bvh_lights,
                              state.num_lights, state.max_depth, refs.regularize,
                              state.sobol_rng, refs.sample_idx,
                              refs.camera, refs.samples_per_pixel, state.rr_depth),
                             Mantle.DeviceRange(q.size; max = Int(q.capacity));
                             group = DEFAULT_WORKGROUPSIZE)
        end
    end
end

"""Transmittance along the shadow rays the medium direct lighting queued."""
function shadow_pass!(g, state::VolPathState, refs)
    Mantle.compute!(g, "shadow") do p
        use!(p, state.shadow_queue; read = true)
        accumulates!(p, state.pixel_L)
        Mantle.dispatch!(p, workqueue_map_kernel!,
                         (vp_trace_shadow_rays_kernel!, state.shadow_queue,
                          state.pixel_L, state.rgb2spec_table, refs.accel,
                          refs.media_interfaces, refs.media, refs.materials),
                         Mantle.DeviceRange(state.shadow_queue.size;
                                            max = Int(state.shadow_queue.capacity));
                         group = DEFAULT_WORKGROUPSIZE)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# The graphs
# ─────────────────────────────────────────────────────────────────────────────

"""
One bounce, appended to `g`.

Which stages exist is a property of the scene, not a runtime branch: a scene
without media never fills the medium queues, and on the hardware per-material
chit path the closest-hit shaders have already shaded the surface by the time
the trace returns. A stage that could only ever dispatch over an empty queue is
left out rather than run for nothing — which is also why the shadow pass is
gated: medium direct lighting is its only producer.
"""
function round_passes!(g, state::VolPathState, refs, cur::WorkQueue, nxt::WorkQueue;
                       chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    reset_pass!(g, state, nxt)
    trace_pass!(g, refs.accel[], state, refs, cur, nxt)
    if has_media
        medium_sample_pass!(g, state, refs, nxt)
        has_lights && medium_dl_pass!(g, state, refs)
        medium_scatter_pass!(g, state, refs, nxt)
    end
    has_lights && escaped_pass!(g, state, refs)
    if !chit_owns_surface
        has_lights && emitters_pass!(g, state, refs)
        shade_pass!(g, state, refs, nxt)
    end
    has_media && has_lights && shadow_pass!(g, state, refs)
    return g
end

"""
    rounds_passes!(g, state, refs, n; …)

`n` consecutive bounces in ONE graph, starting from ray queue A.

A round is not independent of the round before it — it drains the queue that one
filled — so this is not about finding parallelism between them. It is about how
much the host has to say per sample. A plan is recorded when it runs, so a plan
per round means `max_depth` recordings, `max_depth` submissions and `max_depth`
walks over the pass list; `n` rounds in one plan means one of each per `n`.

What makes it expressible at all is that nothing in a round is decided on the
host. Every stage sizes itself off a device-resident count, so a recording is
valid whatever the ray population turns out to be — including zero, where every
dispatch is a no-op and the round costs its commands and nothing else.

The queues alternate inside the graph rather than between plans, which is why
there is one of these and not two: the ping-pong is now an implementation detail
of the recording. `n` rounds starting at A end at A when `n` is even, which is
what lets the same plan run back to back.
"""
function rounds_passes!(g, state::VolPathState, refs, n::Integer;
                        chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    cur, nxt = state.ray_queue_a, state.ray_queue_b
    for _ in 1:n
        round_passes!(g, state, refs, cur, nxt;
                      chit_owns_surface, has_media, has_lights)
        cur, nxt = nxt, cur
    end
    return g
end


"""
The head of a sample: clear the film, empty the ray queue, generate camera rays.

The clear and the reset share a pass because they touch disjoint memory, so
nothing orders them against each other — the fact the `concurrent_dispatch_group`
around them was there to assert.
"""
function setup_passes!(g, state::VolPathState, refs, cur::WorkQueue)
    Mantle.compute!(g, "clear") do p
        Mantle.use(p, state.pixel_L; write = true)
        Mantle.use(p, cur.size; write = true)
        Mantle.dispatch!(p, vp_clear_kernel!, (state.pixel_L, 0f0),
                         length(state.pixel_L))
        Mantle.dispatch!(p, zero_size_counters_kernel!, ((cur.size,),), 1; group = 1)
    end
    Mantle.compute!(g, "camera") do p
        use!(p, cur; read = true, write = true)
        Mantle.use(p, state.wavelengths_per_pixel; write = true)
        Mantle.use(p, state.pdf_per_pixel; write = true)
        Mantle.use(p, state.filter_weight_per_pixel; write = true)
        Mantle.dispatch!(p, vp_generate_camera_rays_kernel!,
                         (cur, state.wavelengths_per_pixel, state.pdf_per_pixel,
                          state.filter_weight_per_pixel, state.height,
                          refs.camera, refs.camera_needs_time, refs.camera_needs_lens,
                          refs.sample_idx, refs.initial_medium,
                          refs.filter_params, refs.filter_sampler, state.sobol_rng),
                         Int(state.width) * Int(state.height))
    end
    return g
end


"""This sample's spectral radiance, weighted into the RGB accumulators."""
function accumulate_passes!(g, state::VolPathState, refs)
    n_pixels = Int(state.width) * Int(state.height)
    Mantle.compute!(g, "accumulate") do p
        Mantle.use(p, state.pixel_L; read = true)
        Mantle.use(p, state.wavelengths_per_pixel; read = true)
        Mantle.use(p, state.pdf_per_pixel; read = true)
        Mantle.use(p, state.filter_weight_per_pixel; read = true)
        Mantle.use(p, state.pixel_rgb; read = true, write = true)
        Mantle.use(p, state.pixel_weight_sum; read = true, write = true)
        Mantle.dispatch!(p, vp_accumulate_to_rgb_kernel!,
                         (state.pixel_rgb, state.pixel_weight_sum, state.pixel_L,
                          state.wavelengths_per_pixel, state.pdf_per_pixel,
                          state.filter_weight_per_pixel,
                          state.cie_table.cie_x, state.cie_table.cie_y,
                          state.cie_table.cie_z, Int32(n_pixels),
                          refs.max_component_value, state.output_matrix,
                          state.imaging_ratio),
                         n_pixels)
    end
    return g
end


"""The divide that turns the accumulators into the picture."""
function finalize_passes!(g, state::VolPathState, framebuffer)
    n_pixels = Int(state.width) * Int(state.height)
    Mantle.compute!(g, "finalize") do p
        Mantle.use(p, state.pixel_rgb; read = true)
        Mantle.use(p, state.pixel_weight_sum; read = true)
        Mantle.use(p, framebuffer; write = true)
        Mantle.dispatch!(p, vp_finalize_film_kernel!,
                         (framebuffer, state.pixel_rgb, state.pixel_weight_sum,
                          state.width, state.height),
                         n_pixels)
    end
    return g
end

finalize_graph(dev, state::VolPathState, framebuffer) =
    finalize_passes!(Mantle.Graph(dev), state, framebuffer)

"""
    sample_graph(dev, state, refs, max_depth; …) -> Graph

A whole sample as one graph: setup, the bounce loop, accumulate.

The loop is a `Mantle.repeat!` gated on the ray queue's own live count. That is
the whole reason this can be one graph — the host neither reads that count nor
decides how many rounds run, so there is nothing to break the recording up for.

**The body is TWO rounds, and two is the minimum rather than a tuning knob.** A
round reads one ray queue and writes the other, so an odd body would leave the
queues swapped and the next iteration would read the wrong one; two returns them
to where it found them, which is what makes one recording valid for every
iteration. It also fixes the overshoot: the loop can only stop on an even
boundary, so at most one dead round runs after the rays are gone.

This replaces a host loop that drained the pipeline every eight rounds to read
the same counter. Eight was that drain's amortisation constant — `live_rounds/8`
stalls per sample, bought with up to seven dead rounds of overshoot — and with
the test on the device there is nothing left to amortise.

`finalize` is not here. It is one dispatch, and `render!` takes a kwarg to skip
it between samples: a per-call decision, which is the one thing a single
recording cannot express.
"""
function sample_graph(dev, state::VolPathState, refs, max_depth::Int32;
                      chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    a = state.ray_queue_a
    g = Mantle.Graph(dev)
    setup_passes!(g, state, refs, a)
    d = Int(max_depth)
    if d >= 2
        Mantle.repeat!(g, d ÷ 2; while_nonzero = a.size) do _
            rounds_passes!(g, state, refs, 2;
                           chit_owns_surface, has_media, has_lights)
        end
    end
    # An odd depth has one round the pairs cannot cover. It runs unconditionally:
    # a round on an empty queue is every stage sizing itself to zero work, which
    # is what makes the loop legal in the first place.
    isodd(d) && rounds_passes!(g, state, refs, 1;
                               chit_owns_surface, has_media, has_lights)
    accumulate_passes!(g, state, refs)
    return g
end

# ─────────────────────────────────────────────────────────────────────────────
# Building and keeping them
# ─────────────────────────────────────────────────────────────────────────────

"""
What the compiled plans were built against.

A plan bakes the resources its passes name and the types its `Ref`s hold, so
each of these is a reason to build new ones rather than reuse: a different film,
a different set of material types, a scene that grew its first medium. The
per-render *values* are not here — that is what the `Ref`s are for.
"""
struct PlanKey
    refs::DataType
    framebuffer::Any
    per_material_queue::Any
    chit_owns_surface::Bool
    has_media::Bool
    has_lights::Bool
    max_depth::Int32
end

# Identity for the resources, equality for the flags — a `LavaArray` compared
# with `==` would read the device back.
Base.:(==)(a::PlanKey, b::PlanKey) =
    a.refs === b.refs &&
    a.framebuffer === b.framebuffer &&
    a.per_material_queue === b.per_material_queue &&
    a.chit_owns_surface == b.chit_owns_surface &&
    a.has_media == b.has_media &&
    a.has_lights == b.has_lights &&
    a.max_depth == b.max_depth

"""
The compiled plans for one `(integrator state, scene shape, film)`, and the
`Ref`s a run writes its per-sample values into.

Two plans, because a sample is two things: everything that produces the picture,
and the divide that writes it out. `sample` holds setup, the whole bounce loop
and accumulate; it used to be four plans driven by a host `while` loop, which
existed only so the host could read the ray count between chunks.
"""
struct VPPlans{R}
    key::PlanKey
    refs::R
    sample::Mantle.Plan
    finalize::Mantle.Plan
end

"""
Whether `build_plans` records its plans up front rather than leaving the first
`run!` to do it.

`false` by default, and it no longer selects between two modes: `Mantle.run!`
records on the first run of any plan it can record, so this only moves WHEN that
happens. Kept because "record before the timing loop" is the difference between
a first sample that includes the recording and one that does not, which is what
made every earlier comparison of the two modes hard to read.
"""
const RECORD_PLANS = Ref(false)


"""
    build_plans(...) -> VPPlans

Compile the sample. Called when the key changes, which for a still scene is
once.
"""
function build_plans(backend, state::VolPathState, framebuffer, refs, max_depth::Int32;
                     chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    dev = mantle_device(backend)
    key = PlanKey(typeof(refs), framebuffer, state.per_material_queue,
                  chit_owns_surface, has_media, has_lights, max_depth)
    plans = VPPlans(key, refs,
                    Mantle.Plan(sample_graph(dev, state, refs, max_depth;
                                             chit_owns_surface, has_media, has_lights)),
                    Mantle.Plan(finalize_graph(dev, state, framebuffer)))
    # There is one mode now. `Mantle.Plan` turns the graph into passes, barriers
    # and pipelines; recording writes the command buffers, and `run!` does it on
    # the first run if nothing else has. This only moves that cost out of the
    # first sample.
    #
    # The old comparison — interpreted against compiled — measured 2-3 % in
    # favour of interpreted on an RTX 4000 Ada on 2026-08-31 (crown -2.2 %,
    # bunny_cloud -1.0 %, killeroo_gold -3.3 %, materials -2.0 %, black_hole
    # +3.5 %), and the other way on the driver before it. The open candidate for
    # that gap was `SIMULTANEOUS_USE`, which a replayed command buffer had to be
    # begun with; a recording is begun with NO flags now, because there is one
    # per argument slot and the ring guarantees the previous submission of that
    # exact buffer has completed. Worth re-measuring rather than believing —
    # neither number is comparable across the change.
    if RECORD_PLANS[]
        for p in allplans(plans)
            Mantle.record!(p)
        end
    end
    return plans
end

"""Every plan of a sample, in the order a sample runs them."""
allplans(p::VPPlans) = (p.sample, p.finalize)

# On recording, which `build_plans` above can do up front.
#
# Recording writes the command buffers once; a run submits them. The host saving
# is real — a round's recording is 0.101 ms and submitting it 0.0058 ms — and it
# used to be swallowed whole by how a replay reached the queue. `replay!`
# submitted on its own, behind a semaphore wait on the previous replay, so
# replays were serialised against each other: an ordering that one recording
# expresses with an intra-submission barrier became a GPU round-trip between
# submissions. Measured then, paired and interleaved in one session: at a chunk
# of 8 rounds baking COST 5.2 % on medium_null, and only at a chunk of 64 — the
# whole sample in one plan, hence one replay — did it win 14.5 %.
#
# That is gone. A recording is appended to the batch the host is already building
# and leaves in the same `vkQueueSubmit2`, so there is no wait and no extra
# submission. Re-measured 2026-08-31 on an RTX 4000 Ada, every plan below baked,
# paired in one session against the same integrator:
#
#     materials  1200x900   10 spp  depth 50    0.744 s -> 0.717 s   1.037x
#     crown      1000x1400  16 spp  depth 100   2.226 s -> 2.130 s   1.045x
#
# Both bit identical to the interpreted render, at every sample count from 1 to 8
# and at 10 and 16. So the chunk stays at 8 — the early exit is worth far more
# than either number.
#
# Four things had to be right before any of this could be compared, and each was
# wrong at some point, so each is worth knowing: `bake!` used to RUN the plan as
# it captured it (the accumulate pass would contribute a spurious sample); a
# baked plan replayed the arguments it captured unless the caller remembered
# `Mantle.rebind!`; `rebind!` could not reach a `custom!` pass's arguments, which
# is what the hardware RT trace used to be; and a recording belongs to the
# argument slot it was captured in, which `bake!` got wrong for any plan that had
# already run. All four are fixed and pinned by tests in Mantle, and three of the
# four are unreachable now — `run!` records, rebinds and rotates for itself, and
# `custom!` is gone.


"""
    ensure_plans!(state, film, backend, scene parts..., per-render values...) -> refs

The plans for this render, built if the key moved and reused otherwise, with
every per-render value written into the `Ref`s the compiled dispatches read.
Returns the refs so the caller can set the sample index it is about to render.

The values are written on the cache hit *and* the miss: a rebuild takes them
from its arguments, and a reuse has to overwrite last call's.
"""
function ensure_plans!(state::VolPathState, film::Film, backend,
                       accel, media_interfaces, media, materials, lights,
                       camera, camera_needs_time::Bool, camera_needs_lens::Bool,
                       initial_medium, filter_params, filter_sampler,
                       regularize::Bool, samples_per_pixel::Int32,
                       max_component_value::Float32, max_depth::Int32,
                       sample_idx::Int32;
                       chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    refs = render_refs(accel, media_interfaces, media, materials, lights,
                       camera, camera_needs_time, camera_needs_lens,
                       initial_medium, filter_params, filter_sampler,
                       regularize, samples_per_pixel, max_component_value)
    key = PlanKey(typeof(refs), film.framebuffer, state.per_material_queue,
                  chit_owns_surface, has_media, has_lights, max_depth)
    plans = state.plans
    if plans === nothing || plans.key != key
        # The old plans go back without a wait. Their recordings may well still
        # be in flight; `Mantle.free!` retires the regions rather than releasing
        # them, so the pool hands those bytes on only once the device says so.
        # This used to be `KA.synchronize(backend)` first.
        plans === nothing || free!(plans)
        # The sample index BEFORE the build, because the build BAKES: a capture
        # takes the arguments as they are, and one taken with the previous
        # sample's index renders that sample again.
        refs.sample_idx[] = sample_idx
        state.plans = build_plans(backend, state, film.framebuffer, refs, max_depth;
                                  chit_owns_surface, has_media, has_lights)
        return state.plans.refs
    end
    r = plans.refs
    r.accel[] = accel
    r.media_interfaces[] = media_interfaces
    r.media[] = media
    r.materials[] = materials
    r.lights[] = lights
    r.camera[] = camera
    r.camera_needs_time[] = camera_needs_time
    r.camera_needs_lens[] = camera_needs_lens
    r.initial_medium[] = initial_medium
    r.filter_params[] = filter_params
    r.filter_sampler[] = filter_sampler
    r.regularize[] = regularize
    r.samples_per_pixel[] = samples_per_pixel
    r.max_component_value[] = max_component_value
    r.sample_idx[] = sample_idx
    return r
end

"""Give every plan's pool regions back. Explicit, like every other release
here — see the `sync!`/`free!` contract."""
free!(plans::VPPlans) = (foreach(Mantle.free!, allplans(plans)); nothing)
