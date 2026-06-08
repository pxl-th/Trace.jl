# Unified WorkQueue for wavefront integrators
# GPU-compatible thread-safe work queue with optional SOA layout
#
# This replaces both PWWorkQueue and VPWorkQueue with a single implementation.

using KernelAbstractions
using KernelAbstractions: @kernel, @index
import KernelAbstractions as KA
using Atomix: @atomic
using StructArrays
using Adapt
import Lava: LavaBackend, concurrent_dispatch_group

# ============================================================================
# SOA/AOS Array Allocation (following pbrt-v4's SOA pattern)
# ============================================================================

# Trait: should this type be decomposed into SOA?
# Override this for work item types that benefit from SOA layout.
# Nested types (Ray, Vec3f, Spectrum, etc.) stay flat since they're small
# and accessed as units. The SOA benefit comes from coalesced access when
# threads read the same field across different work items.
should_use_soa(::Type{T}) where T = false

"""
    allocate_array(backend, T, n; soa=false)

Allocate array with AOS (soa=false) or SOA (soa=true) layout.
Both support identical indexing: arr[i] returns T, arr[i] = val stores T.
"""
function allocate_array(backend, ::Type{T}, n::Integer; soa::Bool=false) where T
    soa ? allocate_soa(backend, T, n) : KA.allocate(backend, T, n)
end

function allocate_soa(backend, ::Type{T}, n::Integer) where T
    if !should_use_soa(T)
        return KA.allocate(backend, T, n)
    end
    if fieldcount(T) > 0
        fnames = fieldnames(T)
        ftypes = fieldtypes(T)
        components = NamedTuple{fnames}(
            ntuple(i -> allocate_soa(backend, ftypes[i], n), length(fnames))
        )
        return StructArray{T}(components)
    end
    return KA.allocate(backend, T, n)
end

# ============================================================================
# WorkQueue - Unified GPU Work Queue
# ============================================================================

"""
    WorkQueue{T, V, S}

A GPU-compatible work queue that stores items of type T.
Uses atomic operations for thread-safe push operations.

The queue stores items in a pre-allocated buffer and uses an atomic
counter to track the current size. Supports both AOS and SOA layouts.

# Fields
- `items::V`: Pre-allocated array of work items
- `size::S`: Single-element array for atomic counter
- `capacity::Int32`: Maximum number of items

# Example
```julia
# Create a queue on GPU backend
queue = WorkQueue{MyWorkItem}(backend, 1024)

# In a kernel, push items atomically
idx = push!(queue, item)

# Check if push succeeded (within capacity)
if idx <= length(queue.items)
    # item was stored at queue.items[idx]
end
```
"""
struct WorkQueue{T, V <: AbstractVector{T}, S <: AbstractVector{Int32}}
    items::V
    size::S      # Single-element array for atomic operations
    capacity::Int32
end

"""
    free!(queue::WorkQueue)

Release GPU memory held by the work queue's items and size arrays.
Does **not** synchronize — caller must ensure the GPU is idle (see the
sync!/free! contract in Raycore and Hikari).
"""
function free!(queue::WorkQueue)
    # SOA queues store `items` as a `StructArray{T}` whose components are
    # LavaArrays; `finalize(::StructArray)` is a no-op and leaves the
    # component LavaArrays alive until Julia GC runs.  AOS queues store
    # `items` as a single LavaArray directly.  Walk in either case.
    _finalize_items!(queue.items)
    finalize(queue.size)
    return nothing
end
_finalize_items!(items) = finalize(items)
function _finalize_items!(items::StructArray)
    for c in StructArrays.components(items)
        _finalize_items!(c)
    end
end

"""
    WorkQueue{T}(backend, capacity; soa=should_use_soa(T))

Create a new work queue with the given capacity on the specified backend.

# Arguments
- `backend`: KernelAbstractions backend (e.g., `CPU()`, `CUDABackend()`, `ROCBackend()`)
- `capacity`: Maximum number of items the queue can hold
- `soa`: If true, use Structure-of-Arrays layout for better GPU memory coalescing.
  Defaults to `should_use_soa(T)` so flagged item types automatically get the
  faster layout without every caller having to remember the kwarg. (Before
  this default was wired up, the trait was set on `VPRayWorkItem` and friends
  but never actually read — every queue was AOS regardless.)
"""
function WorkQueue{T}(backend, capacity::Integer; soa::Bool=should_use_soa(T)) where T
    items = allocate_array(backend, T, capacity; soa=soa)
    size = KA.allocate(backend, Int32, 1)
    KA.fill!(size, Int32(0))
    WorkQueue{T, typeof(items), typeof(size)}(items, size, Int32(capacity))
end

# ============================================================================
# Queue Operations
# ============================================================================

function Base.length(queue::WorkQueue)
    s = Array(queue.size)
    return Int(s[1])
end

@inline function Base.push!(queue::WorkQueue{T}, item::T) where T
    # Atomically increment size and get index
    idx = @atomic queue.size[1] += Int32(1)
    # Only store if within bounds
    if idx <= length(queue.items)
        @inbounds queue.items[idx] = item
    end
    return idx
end

@propagate_inbounds function Base.getindex(queue::WorkQueue, idx::Integer)
    return queue.items[idx]
end

@propagate_inbounds function Base.setindex!(queue::WorkQueue{T}, item::T, idx::Integer) where T
    queue.items[idx] = item
    return item
end

function Base.empty!(queue::WorkQueue)
    fill!(queue.size, Int32(0))
    return queue
end


# ============================================================================
# Adapt.jl Integration for GPU Kernels
# ============================================================================

"""
    Adapt.adapt_structure(backend, queue::WorkQueue)

Adapt WorkQueue for use inside GPU kernels. This converts the host-side
arrays (e.g., CLArray, CuArray) to device-compatible representations
(e.g., CLDeviceArray, CuDeviceArray) that can be used inside kernels.

This allows passing the entire WorkQueue to a kernel instead of
passing items and size arrays separately.
"""
function Adapt.adapt_structure(backend, queue::WorkQueue)
    WorkQueue(
        Adapt.adapt(backend, queue.items),
        Adapt.adapt(backend, queue.size),
        queue.capacity
    )
end

# ============================================================================
# Map Operations for GPU Kernel Execution
# ============================================================================
@kernel function workqueue_map_kernel!(f, queue, args...)
    i = @index(Global)
    if i <= queue.size[1]
        @inbounds f(queue.items[i], args...)
    end
end

# Default workgroupsize=256 gives ~14% speedup on CUDA (Ampere) vs auto-selection.
# Static workgroupsize helps GPU compilers optimize register allocation and enables
# more concurrent blocks per SM.
const DEFAULT_WORKGROUPSIZE = 256

"""
    gpu_ndrange(backend, size_buf)

Get the ndrange for dispatching over a queue's GPU-resident size buffer.
Default: CPU readback (works on AMDGPU, CUDA, CPU).
Lava overrides this to return the GPU array directly for indirect dispatch (no flush).
"""
# Clamp to 1 (not 0) because ndrange=0 crashes on some backends (AMDGPU).
# The kernel's bounds check (idx > queue_size) handles the empty case.
# Backends supporting indirect dispatch (Lava) return the GPU array directly,
# avoiding GPU->CPU sync. Others fall back to CPU readback.
function gpu_ndrange(backend, size_buf)
    if backend isa LavaBackend
        return size_buf
    end
    return max(Int(Array(size_buf)[1]), 1)
end

function Base.foreach(f, queue::WorkQueue, args...; workgroupsize=DEFAULT_WORKGROUPSIZE)
    backend = KA.get_backend(queue.items)
    kernel! = workqueue_map_kernel!(backend, workgroupsize)
    kernel!(f, queue, args...; ndrange=gpu_ndrange(backend, queue.size))
    return nothing
end

# ============================================================================
# MultiTypeWorkQueue — one WorkQueue per concrete item type
# ============================================================================
#
# Mirrors pbrt-v4's per-material-type queue pattern (MaterialEvalQueue<T> in
# wavefront/workitems.h). One `WorkQueue{ItemFor[T]}` per concrete type T in
# the scene. The trace kernel routes each work item into the queue for its
# concrete type via `with_index` dispatch; the consumer drains *all* of them
# with `foreach_type`, which lowers to one indirect dispatch per type into
# the active Lava command buffer.
#
# Each per-type kernel is fully monomorphised by Julia (no `with_index`
# switch inside the kernel), so the resulting SPIR-V is small and the warps
# never diverge on material-type branches.
#
# On Lava the N dispatches naturally serialise via the existing per-dispatch
# barriers. Wrapping the loop in `Lava.concurrent_dispatch_group()` (see
# Lava commit eefc75c) suppresses the inter-dispatch barriers so independent
# per-type kernels run concurrently on idle SMs — verified to give up to
# ~3× wall-clock speedup on small dispatches that don't saturate the GPU.

"""
    MultiTypeWorkQueue{Qs <: Tuple}

Heterogeneous tuple of `WorkQueue`s, one per concrete item type. Iteration
order matches the type-tuple order. Compile-time-known number of queues, so
`foreach_type` unrolls cleanly via `Base.foreach(::Tuple)` (which lowers to
`afoldl`) with full type stability per arm.

# Fields
- `queues::Qs`: NTuple of WorkQueue, one per concrete item type
"""
struct MultiTypeWorkQueue{Qs <: Tuple}
    queues::Qs
end

"""
    MultiTypeWorkQueue(item_types::Tuple, capacity, backend; soa=false)

Build a MultiTypeWorkQueue with one `WorkQueue{T}(backend, capacity)` per `T`
in `item_types`.

```julia
mtwq = MultiTypeWorkQueue((HitWorkA, HitWorkB, HitWorkC), 1024, backend)
```
"""
function MultiTypeWorkQueue(item_types::Tuple, capacity::Integer, backend; soa::Bool=false)
    qs = map(T -> WorkQueue{T}(backend, capacity; soa=soa), item_types)
    return MultiTypeWorkQueue(qs)
end

function free!(mtwq::MultiTypeWorkQueue)
    foreach(free!, mtwq.queues)
    return nothing
end

Base.empty!(mtwq::MultiTypeWorkQueue) = (foreach(empty!, mtwq.queues); mtwq)

# Adapt walks into the tuple so each per-queue Adapt.adapt_structure runs
# and the kernel sees device-side arrays.
function Adapt.adapt_structure(backend, mtwq::MultiTypeWorkQueue)
    MultiTypeWorkQueue(map(q -> Adapt.adapt(backend, q), mtwq.queues))
end

"""
    foreach_type(kernel!, mtwq::MultiTypeWorkQueue, args...; workgroupsize=DEFAULT_WORKGROUPSIZE)

Dispatch `kernel!` once per queue in `mtwq`, indirect-dispatched on each
queue's GPU-resident size buffer. Julia unrolls the tuple loop at compile
time (via `Base.foreach(::Tuple) → afoldl`) and the kernel is monomorphised
per concrete item type, so each arm compiles to a separate small SPIR-V
module with no `with_index` switch inside.

Per-dispatch barriers between the per-type kernels serialise them on the
GPU. Wrap the call in `Lava.concurrent_dispatch_group(...) do ... end` to
let them overlap when they're independent (different output queues, or
shared output via atomic-claimed slots).
"""
@inline function foreach_type(kernel!, mtwq::MultiTypeWorkQueue, args...;
                              workgroupsize=DEFAULT_WORKGROUPSIZE)
    foreach(q -> foreach(kernel!, q, args...; workgroupsize=workgroupsize), mtwq.queues)
    return nothing
end

# ============================================================================
# Convenience Aliases
# ============================================================================

# For backwards compatibility during migration
const GPUWorkQueue = WorkQueue
