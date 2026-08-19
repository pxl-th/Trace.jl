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
function use!(p, x; read::Bool = false, write::Bool = false)
    for b in devicebuffers!(Any[], x)
        Mantle.use(p, b; read = read, write = write)
    end
    return nothing
end

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
    use!(p, state.pixel_L; read = true, write = true)
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
        use!(p, state.pixel_L; read = true, write = true)
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
        use!(p, state.pixel_L; read = true, write = true)
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
        use!(p, state.pixel_L; read = true, write = true)
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
        use!(p, state.pixel_L; read = true, write = true)
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
        use!(p, state.pixel_L; read = true, write = true)
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
    rounds_graph(dev, state, refs, n; …)

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
function rounds_graph(dev, state::VolPathState, refs, n::Integer;
                      chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    g = Mantle.Graph(dev)
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
function setup_graph(dev, state::VolPathState, refs, cur::WorkQueue)
    g = Mantle.Graph(dev)
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
function accumulate_graph(dev, state::VolPathState, refs)
    n_pixels = Int(state.width) * Int(state.height)
    g = Mantle.Graph(dev)
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
function finalize_graph(dev, state::VolPathState, framebuffer)
    n_pixels = Int(state.width) * Int(state.height)
    g = Mantle.Graph(dev)
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

`chunk` is [`EXIT_CHECK_INTERVAL`](@ref) bounces in one plan and `tail` is
whatever is left over, or `nothing` when the depth divides evenly. Both start
from ray queue A, so the loop runs `chunk` back to back and `tail` once — the
chunk length is even, which is what makes that legal.
"""
struct VPPlans{R}
    key::PlanKey
    refs::R
    setup::Mantle.Plan
    chunk::Union{Nothing,Mantle.Plan}
    chunk_rounds::Int
    tail::Union{Nothing,Mantle.Plan}
    tail_rounds::Int
    accumulate::Mantle.Plan
    finalize::Mantle.Plan
end

"""
    build_plans(...) -> VPPlans

Compile the sample. Called when the key changes, which for a still scene is
once.
"""
function build_plans(backend, state::VolPathState, framebuffer, refs, max_depth::Int32;
                     chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    dev = mantle_device(backend)
    a = state.ray_queue_a
    key = PlanKey(typeof(refs), framebuffer, state.per_material_queue,
                  chit_owns_surface, has_media, has_lights, max_depth)
    nchunk, ntail = chunking(max_depth)
    plan(n) = n == 0 ? nothing :
        Mantle.Plan(rounds_graph(dev, state, refs, n;
                                 chit_owns_surface, has_media, has_lights))
    plans = VPPlans(key, refs,
                    Mantle.Plan(setup_graph(dev, state, refs, a)),
                    plan(nchunk), nchunk,
                    plan(ntail), ntail,
                    Mantle.Plan(accumulate_graph(dev, state, refs)),
                    Mantle.Plan(finalize_graph(dev, state, framebuffer)))
    return plans
end

"""Every plan of a sample, in the order a sample runs them."""
allplans(p::VPPlans) = filter(!isnothing,
                              (p.setup, p.chunk, p.tail, p.accumulate, p.finalize))

# NOT baked, and the reason is measured rather than assumed.
#
# `Mantle.bake!` replays a capture instead of re-recording, and the host saving
# is real — a round's recording is 0.101 ms and its replay 0.0058 ms. But
# `Lava.replay!` waits on a semaphore for the PREVIOUS replay, so replays are
# serialised against each other: an ordering that a single recording expresses
# with an intra-submission barrier becomes a GPU round-trip between submissions.
# That is free for what capture was built for — one plan replayed once per
# inference step — and it is not free for a renderer that replays a chunk four
# times a sample and thirty-two samples a frame.
#
# Paired and interleaved, unbaked against baked, in one session: at a chunk of 8
# rounds baking costs +5.2 % on medium_null, and at a chunk of 64 — the whole
# sample in one plan, hence ONE replay — it wins 14.5 %. Which is the
# serialisation, changing sign exactly where the model says it should. The
# whole-sample plan is not the way out: it gives up the early exit, worth far
# more than 14.5 % on any scene whose rays die early.
#
# Three things had to be right before that comparison meant anything, and each
# is a trap worth knowing: `bake!` RUNS the plan as it captures it (so the
# accumulate pass would contribute a spurious sample); a baked plan replays the
# arguments it captured unless `Mantle.rebind!` writes new ones; and `rebind!`
# cannot reach a `custom!` pass's arguments, which is the hardware RT trace.


"""
    chunking(max_depth) -> (chunk_rounds, tail_rounds)

How a sample's bounces split into plans.

The whole depth in one plan would be the fewest submissions, and it would cost
the early exit — which is the thing that makes a deep scene affordable, since
rays die long before `max_depth` and every dead round still costs its commands.
So the split is the exit's own granularity: it already only looks every
[`EXIT_CHECK_INTERVAL`](@ref) rounds, so a plan of exactly that many gives up
nothing it was not giving up already, and the host speaks once per eight bounces
instead of once per bounce.

The chunk length has to be EVEN, or the ray queues would not be back where the
plan expects them for the next run. `EXIT_CHECK_INTERVAL` is 8; a tail that is
odd is fine, because a tail runs last.
"""
function chunking(max_depth::Integer)
    d = Int(max_depth)
    k = Int(EXIT_CHECK_INTERVAL)
    isodd(k) && throw(ArgumentError(
        "EXIT_CHECK_INTERVAL must be even: a chunk of rounds has to leave the ray " *
        "queues where the next chunk expects them, and an odd count leaves them " *
        "swapped."))
    d >= k ? (k, d % k) : (0, d)
end

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
        # Before the old plans' regions go back: they were recorded against
        # this device and may still be in flight.
        if plans !== nothing
            KA.synchronize(backend)
            free!(plans)
        end
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
