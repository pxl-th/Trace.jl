const SAH = Val{:SAH}
const HLBVH = Val{:HLBVH}
const BVHSplitMethods = Union{SAH, HLBVH}

abstract type AccelPrimitive <: Primitive end

struct BVHPrimitiveInfo
    primitive_number::UInt32
    bounds::Bounds3
    centroid::Point3f0

    function BVHPrimitiveInfo(primitive_number::Integer, bounds::Bounds3)
        new(primitive_number, bounds, 0.5f0 * bounds.p_min + 0.5f0 * bounds.p_max)
    end
end

struct BVHNode
    bounds::Bounds3
    children::Tuple{Maybe{BVHNode}, Maybe{BVHNode}}
    split_axis::UInt8
    offset::UInt32
    n_primitives::UInt32

    function BVHNode(offset::Integer, n_primitives::Integer, bounds::Bounds3)
        new(bounds, (nothing, nothing), 0, offset, n_primitives)
    end
    function BVHNode(axis::Integer, left::BVHNode, right::BVHNode)
        new(left.bounds ∪ right.bounds, (left, right), axis, 0, 0)
    end
end

struct BVHAccel{M <: BVHSplitMethods} <: AccelPrimitive
    primitives::Vector{P} where P <: Primitive
    max_node_primitives::UInt8
    root::BVHNode

    function BVHAccel{M}(
        primitives::Vector{P}, max_node_primitives::Integer = 1,
    ) where M <: BVHSplitMethods where P <: Primitive
        M == HLBVH && error("HLBVH method not implemented yet.")

        max_node_primitives = min(255, max_node_primitives)
        length(primitives) == 0 && return new{M}(primitives, max_node_primitives)

        primitives_info = [
            BVHPrimitiveInfo(i, p |> world_bound)
            for (i, p) in enumerate(primitives)
        ]

        total_nodes = 0 |> Ref
        ordered_primitives = Vector{P}(undef, 0)
        root = _init_bvh(
            primitives, primitives_info, 1, primitives |> length,
            total_nodes, ordered_primitives, max_node_primitives,
        )

        new{M}(ordered_primitives, max_node_primitives, root)
    end
end

mutable struct BucketInfo
    count::UInt32
    bounds::Bounds3
end

function _init_bvh(
    primitives::Vector{P}, primitives_info::Vector{BVHPrimitiveInfo},
    from::Integer, to::Integer, total_nodes::Ref{Int64},
    ordered_primitives::Vector{P}, max_node_primitives::Integer,
) where P <: Primitive
    # Compute bounds for all primitives in BVH node.
    total_nodes[] += 1
    bounds = mapreduce(i -> primitives_info[i].bounds, ∪, from:to)
    n_primitives = to - from
    if n_primitives == 0
        first_offset = ordered_primitives |> length
        for i in from:to
            push!(ordered_primitives, primitives[primitives_info[i].primitive_number])
        end
        return BVHNode(first_offset, 1, bounds)
    end
    n_primitives += 1
    # Compute bound of primitive centroids, choose split dimension.
    centroid_bounds = mapreduce(i -> primitives_info[i].centroid |> Bounds3, ∪, from:to)
    dim = centroid_bounds |> maximum_extent
    # Create leaf node.
    if centroid_bounds.p_min[dim] == centroid_bounds.p_max[dim]
        first_offset = ordered_primitives |> length
        for i in from:to
            push!(ordered_primitives, primitives[primitives_info[i].primitive_number])
        end
        return BVHNode(first_offset, n_primitives, bounds)
    end
    # Partition primitives into sets and build children.
    if n_primitives <= 2 # Equally-sized subsets.
        mid = (from + to) ÷ 2
        pmid = mid > from ? mid - from + 1 : 1
        partialsort!(@view(primitives_info[from:to]), pmid, by=i -> i.centroid[dim])
    else # Perform Surface-Area-Heuristic partitioning.
        n_buckets = 12
        buckets = [BucketInfo(0, Bounds3()) for _ in 1:n_buckets]
        # Initialize buckets.
        for i in from:to
            b = Int32(floor(n_buckets * offset(centroid_bounds, primitives_info[i].centroid)[dim])) + 1
            (b == n_buckets + 1) && (b -= 1)
            buckets[b].count += 1
            buckets[b].bounds = buckets[b].bounds ∪ primitives_info[i].bounds
        end
        # Compute costs for splitting after each bucket.
        costs = Vector{Float32}(undef, n_buckets - 1)
        for i in 1:(n_buckets - 1)
            it1, it2 = 1:i, (i + 1):(n_buckets - 1)
            s1, s2 = 0, 0
            length(it1) > 0 && (s1 = length(it1) * surface_area(mapreduce(b -> buckets[b].bounds, ∪, it1)))
            length(it2) > 0 && (s2 = length(it2) * surface_area(mapreduce(b -> buckets[b].bounds, ∪, it2)))
            costs[i] = 0.125f0 + (s1 + s2) / surface_area(bounds)
        end
        # Find bucket to split that minimizes SAH metric.
        min_cost_id = costs |> argmin
        leaf_cost = n_primitives
        # Either create leaf or split primitives at selected SAH bucket.
        if !(n_primitives > max_node_primitives || costs[min_cost_id] < leaf_cost)
            first_offset = ordered_primitives |> length
            for i in from:to
                push!(ordered_primitives, primitives[primitives_info[i].primitive_number])
            end
            return BVHNode(first_offset, n_primitives, bounds)
        end
        mid = partition!(primitives_info, from:to, i -> begin
            b = Int32(floor(n_buckets * offset(centroid_bounds, i.centroid)[dim])) + 1
            (b == n_buckets + 1) && (b -= 1)
            b <= min_cost_id
        end)
    end
    BVHNode(
        dim,
        _init_bvh(primitives, primitives_info, from, mid, total_nodes, ordered_primitives, max_node_primitives),
        _init_bvh(primitives, primitives_info, mid + 1, to, total_nodes, ordered_primitives, max_node_primitives),
    )
end

world_bound(bvh::BVHAccel) = bvh.root ≢ nothing ? bvh.root.bounds : Bounds3()
