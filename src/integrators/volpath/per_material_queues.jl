# Per-material-type queue infrastructure (pbrt-v4 wavefront pattern).
#
# Today's `vp_trace_and_shade_kernel!` shades each surface hit inline by
# going through `with_index(materials, idx, ...)` which expands to a switch
# over every concrete material type in the scene's `StaticMultiTypeSet`.
# That makes the kernel large (every material's BSDF code is inlined into
# the switch arms) and divergent (threads with different materials in the
# same warp follow different arms).
#
# pbrt-v4 sidesteps both by splitting the shading queue by concrete material
# type — `MaterialEvalQueue<ConcreteMaterial>` in wavefront/workitems.h —
# and running one shading kernel per type via `ForEachType` (per-type kernel
# launches in surfscatter.cpp:41). Each kernel is templated on a single
# concrete material, so the BSDF code is fully monomorphised; warps process
# threads that all run the same code path.
#
# The pieces below are the Julia analogue:
#
#   * `TypedHit{T}` — a thin wrapper around `VPHitSurfaceWorkItem` that
#     attaches the concrete material type `T` at the type level. The
#     wrapper is zero-cost at runtime; its only purpose is to let Julia's
#     normal multiple dispatch pick a kernel specialisation per `T`.
#
#   * `MultiTypeMaterialQueue` — a `MultiTypeWorkQueue` of `TypedHit{T}` for
#     each `T` in the scene's `StaticMultiTypeSet`. Built once at integrator
#     state init from the materials set's static type tuple.
#
#   * `push_typed_hit!(mtmq, materials, hit)` — uses `with_index` on the
#     materials set ONE time (per hit, at push site) to route into the
#     matching typed queue. The kernel that drains a typed queue never
#     touches `with_index`; it only sees one concrete `TypedHit{T}` type.
#
#   * `material_of_type(materials, T, vec_idx)` — a `@generated` lookup that
#     returns the concrete material instance for a known type `T` and the
#     `vec_idx` carried on the hit. Zero overhead — compiles down to one
#     array load on the data tuple at the slot where the type lives.

using Adapt
import KernelAbstractions as KA
import Raycore

# ─────────────────────────────────────────────────────────────────────────────
# TypedHit{T}: zero-cost wrapper that lifts the material type into the type
# parameter of the work item.
# ─────────────────────────────────────────────────────────────────────────────

"""
    TypedHit{T}

Wrapper around `VPHitSurfaceWorkItem` that attaches the concrete material
type `T` to the item type. Per-material shading kernels dispatch on
`TypedHit{T}` — each `T` produces a separately compiled kernel via Julia's
normal multiple dispatch.

The wrapper has no runtime cost: `sizeof(TypedHit{T}) == sizeof(VPHitSurfaceWorkItem)`.
"""
struct TypedHit{T}
    hit::VPHitSurfaceWorkItem
end

# Adapt walks into the hit so the kernel sees device-side fields.
Adapt.adapt_structure(to, t::TypedHit{T}) where {T} =
    TypedHit{T}(Adapt.adapt(to, t.hit))

# ─────────────────────────────────────────────────────────────────────────────
# material_of_type — compile-time-resolved per-type material lookup.
# ─────────────────────────────────────────────────────────────────────────────

"""
    material_of_type(materials::StaticMultiTypeSet, ::Type{T}, vec_idx::UInt32)

Return the concrete material instance of type `T` at position `vec_idx`
within the type-`T` slot of the materials set. The slot index is looked up
at compile time from the materials set's static type tuple, so this lowers
to a single indexed array load.

Used by per-material shading kernels: the kernel knows its material type
`T` (from the `TypedHit{T}` it processes) and `vec_idx` (from the hit's
`material_idx.vec_idx`), and reads the material directly — no `with_index`
switch.
"""
@inline @generated function material_of_type(
    materials::Raycore.StaticMultiTypeSet{Data},
    ::Type{T},
    vec_idx::UInt32,
) where {Data, T}
    # Find the slot index where eltype matches T
    slot = 0
    for (i, vec_type) in enumerate(Data.parameters)
        if eltype(vec_type) === T
            slot = i
            break
        end
    end
    if slot == 0
        return :(error($("material_of_type: type $T not present in materials set")))
    end
    return quote
        @inbounds materials.data[$slot][vec_idx]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# MultiTypeMaterialQueue — MultiTypeWorkQueue of TypedHit{T} per material slot
# ─────────────────────────────────────────────────────────────────────────────

"""
    MultiTypeMaterialQueue

Alias for `MultiTypeWorkQueue` whose queues hold `TypedHit{T}` items, one
queue per concrete material type in the scene. Built from a materials
`StaticMultiTypeSet` via `build_per_material_queues`.
"""
const MultiTypeMaterialQueue{Qs} = MultiTypeWorkQueue{Qs}

"""
    build_per_material_queues(materials::StaticMultiTypeSet, capacity::Integer, backend)

Build one `WorkQueue{TypedHit{T}}` per concrete material type `T` in the
scene's `materials` set. Queue iteration order matches the materials set's
data-tuple order, which is also what `material_idx.type_idx` indexes into,
so a hit with `type_idx == i` belongs in queue `i`.
"""
function build_per_material_queues(
    materials::Raycore.StaticMultiTypeSet{Data},
    capacity::Integer,
    backend,
) where {Data}
    item_types = map(vec_type -> TypedHit{eltype(vec_type)}, Data.parameters)
    return MultiTypeWorkQueue(tuple(item_types...), capacity, backend)
end

# ─────────────────────────────────────────────────────────────────────────────
# push_typed_hit! — `with_index` once at push site, queue dispatch is then
# data-driven (just an indexed tuple access on the queue side).
# ─────────────────────────────────────────────────────────────────────────────

# Internal: pushes the hit into the correct typed queue. `with_index` picks
# the concrete material type `T`; this helper wraps the hit in `TypedHit{T}`
# and pushes into the matching queue slot.
@inline function _push_to_typed_queue!(
    ::M, materials, mtmq::MultiTypeMaterialQueue, hit::VPHitSurfaceWorkItem,
) where {M}
    # Find the queue slot for material type M at compile time, push the wrapped hit.
    _push_typed_hit_slot!(mtmq, M, hit)
    return nothing
end

# Compile-time slot lookup + push: identical to material_of_type but for queues.
@inline @generated function _push_typed_hit_slot!(
    mtmq::MultiTypeMaterialQueue{Qs},
    ::Type{M},
    hit::VPHitSurfaceWorkItem,
) where {Qs, M}
    # `Qs` is a Tuple{WorkQueue{TypedHit{T1}, ...}, WorkQueue{TypedHit{T2}, ...}, ...}.
    # The WorkQueue's first type parameter IS the item type (T in `WorkQueue{T, V, S}`).
    slot = 0
    for (i, q_type) in enumerate(Qs.parameters)
        item_type = q_type.parameters[1]              # WorkQueue{T, V, S} → T
        if item_type <: TypedHit && item_type.parameters[1] === M
            slot = i
            break
        end
    end
    if slot == 0
        return :(error($("_push_typed_hit_slot!: no queue for material type $M")))
    end
    return quote
        @inbounds push!(mtmq.queues[$slot], TypedHit{M}(hit))
        nothing
    end
end

"""
    push_typed_hit!(mtmq, materials, hit::VPHitSurfaceWorkItem)

Route `hit` into the matching per-material typed queue. Uses `with_index`
on `materials` *once* to pick the concrete material type, then a
compile-time tuple lookup to push into the right queue slot.
"""
@propagate_inbounds function push_typed_hit!(
    mtmq::MultiTypeMaterialQueue, materials, hit::VPHitSurfaceWorkItem,
)
    Raycore.with_index(_push_to_typed_queue!, materials, hit.material_idx, materials, mtmq, hit)
    return nothing
end

"""
    enqueue_after_intersection!(per_material_queue, hit_area_light_queue,
                                materials, hit::VPHitSurfaceWorkItem,
                                arealight_flat_idx::UInt32,
                                triangle_area::Float32, t_hit::Float32)

Mirrors pbrt-v4's `EnqueueWorkAfterIntersection`
(wavefront/intersect.h:48-149). For surface intersections (no medium) we
need two parallel pushes:

* If the hit triangle is part of an area light
  (`arealight_flat_idx > 0`), push a `VPHitAreaLightWorkItem` so the
  dedicated emitter kernel can run MIS for the indirect-ray hit.
* Always push the hit into the per-material typed queue — the material
  kernel will sample direct lighting + continuation rays.

`arealight_flat_idx`, `triangle_area`, `t_hit` are passed as separate
arguments rather than stored on `VPHitSurfaceWorkItem` because they're
emission-MIS-only — pbrt-v4's `MaterialEvalWorkItem` doesn't carry them
either. Keeping the per-material queue items free of these fields saves
12 B per item.
"""
@propagate_inbounds function enqueue_after_intersection!(
    per_material_queue::MultiTypeMaterialQueue,
    hit_area_light_queue::WorkQueue{VPHitAreaLightWorkItem},
    materials,
    hit::VPHitSurfaceWorkItem,
    arealight_flat_idx::UInt32,
    triangle_area::Float32,
    t_hit::Float32,
)
    if arealight_flat_idx > UInt32(0)
        wo = -hit.ray.d
        push!(hit_area_light_queue, VPHitAreaLightWorkItem(
            arealight_flat_idx,
            hit.pi, hit.n, hit.uv, wo,
            hit.lambda, hit.depth,
            hit.beta, hit.r_u, hit.r_l,
            hit.prev_intr_p, hit.prev_intr_n,
            hit.specular_bounce,
            hit.pixel_index,
            triangle_area, t_hit,
        ))
    end
    push_typed_hit!(per_material_queue, materials, hit)
    return nothing
end
