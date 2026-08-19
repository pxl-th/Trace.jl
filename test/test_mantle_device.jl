using Test
using Hikari, Lava, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

# Hikari's memory and Mantle's have to come from the SAME Vulkan device. Mantle
# devices own a `Pool`, so a second `Device(Lava)` would be a second allocator
# over one `VkDevice` — every buffer still valid, every placement decision made
# against half the picture.
#
# `Mantle.Device(Lava)` caches one per `VkContext`, and this pins that Hikari
# goes through that cache rather than around it.
@testset "mantle_device reuses Lava's device" begin
    backend = Lava.LavaBackend()
    d = Hikari.mantle_device(backend)

    @test d === Hikari.mantle_device(backend)          # cached, not rebuilt
    @test d.ctx === Lava.vk_context()                  # the context Hikari renders on
    @test d.bq === Lava.vk_context().default_bq        # and its queue, not a new one
end

@testset "a graph can read Hikari's own buffers" begin
    # The property the port depends on: Mantle cannot ADOPT a foreign buffer
    # into its arena, but a pass can read one. Without this, moving one kernel
    # into a graph would drag in every producer of its inputs.
    backend = Lava.LavaBackend()
    dev = Hikari.mantle_device(backend)
    n = 256
    src = KA.allocate(backend, Float32, n)             # allocated the Hikari way
    KA.fill!(src, 2f0)

    g = Mantle.Graph(dev)
    dst = Mantle.Buffer(dev, zeros(Float32, n))
    Mantle.compute!(g, "read-foreign") do p
        Mantle.dispatch!(p, Hikari.mantle_probe_scale!,
                         (Mantle.use(p, dst; write = true),
                          Mantle.use(p, src; read = true), 3f0), n)
    end
    Mantle.run!(Mantle.Plan(g))
    @test all(==(6f0), Array(Mantle.storage(dst)))
end

# The two things every wavefront stage needs from a graph, on the smallest thing
# that shows them. Both used to be `foreach(f, queue, args...)`: an argument that
# is a STRUCT of device arrays, and an ndrange that is a counter the host never
# reads.
@kernel function mantle_probe_fill_queue!(q, n::Int32)
    i = @index(Global)
    @inbounds if i <= n
        push!(q, Float32(i))
    end
end

@kernel function mantle_probe_drain_queue!(q, out)
    i = @index(Global)
    @inbounds if i <= q.size[1]
        out[i] = q.items[i] * 2f0
    end
end

@testset "a work queue is a dispatch argument, and its count is the ndrange" begin
    backend = Lava.LavaBackend()
    dev = Hikari.mantle_device(backend)
    cap, want = 4096, 777
    q = Hikari.WorkQueue{Float32}(backend, cap)
    out = KA.allocate(backend, Float32, cap)
    KA.fill!(out, 0f0)

    g = Mantle.Graph(dev)
    Mantle.compute!(g, "fill") do p
        Hikari.use!(p, q; write = true)
        Mantle.dispatch!(p, mantle_probe_fill_queue!, (q, Int32(want)), cap)
    end
    Mantle.compute!(g, "drain") do p
        Hikari.use!(p, q; read = true)
        Mantle.dispatch!(p, mantle_probe_drain_queue!,
                         (q, Mantle.use(p, out; write = true)),
                         Mantle.DeviceRange(q.size; max = cap))
    end
    Mantle.run!(Mantle.Plan(g))
    KA.synchronize(backend)

    # The drain covered exactly what the fill pushed. Both halves matter: the
    # queue reached the kernel as device arrays rather than as the host struct,
    # and the second pass sized itself off a count written by the first — which
    # is also the ordering the graph had to derive, since nothing else puts the
    # two passes in that order.
    o = Array(out)
    @test count(!=(0f0), o) == want
    @test sort(o[1:want]) == Float32[2i for i in 1:want]
end

@testset "several device-sized dispatches in one pass" begin
    # What the per-material shading stage is: N queues drained in one pass,
    # each over its own count. They share one fused prepare and one barrier, so
    # a count going astray between them would show up as the wrong number of
    # elements written for that queue alone.
    backend = Lava.LavaBackend()
    dev = Hikari.mantle_device(backend)
    cap, k = 2048, 4
    qs = [Hikari.WorkQueue{Float32}(backend, cap) for _ in 1:k]
    outs = [KA.allocate(backend, Float32, cap) for _ in 1:k]
    foreach(o -> KA.fill!(o, 0f0), outs)
    wants = [500 + i for i in 1:k]

    g = Mantle.Graph(dev)
    Mantle.compute!(g, "fill") do p
        for (q, w) in zip(qs, wants)
            Hikari.use!(p, q; write = true)
            Mantle.dispatch!(p, mantle_probe_fill_queue!, (q, Int32(w)), cap)
        end
    end
    Mantle.compute!(g, "drain") do p
        for (q, o) in zip(qs, outs)
            Hikari.use!(p, q; read = true)
            Mantle.dispatch!(p, mantle_probe_drain_queue!,
                             (q, Mantle.use(p, o; write = true)),
                             Mantle.DeviceRange(q.size; max = cap))
        end
    end
    Mantle.run!(Mantle.Plan(g))
    KA.synchronize(backend)
    @test [count(!=(0f0), Array(o)) for o in outs] == wants
end

@testset "an empty queue dispatches nothing" begin
    # The ordinary case in a bounce loop, not an edge: every stage past the
    # depth where the rays died drains a queue nobody filled.
    backend = Lava.LavaBackend()
    dev = Hikari.mantle_device(backend)
    cap = 256
    q = Hikari.WorkQueue{Float32}(backend, cap)
    out = KA.allocate(backend, Float32, cap)
    KA.fill!(out, 7f0)

    g = Mantle.Graph(dev)
    Mantle.compute!(g, "drain") do p
        Hikari.use!(p, q; read = true)
        Mantle.dispatch!(p, mantle_probe_drain_queue!,
                         (q, Mantle.use(p, out; write = true)),
                         Mantle.DeviceRange(q.size; max = cap))
    end
    Mantle.run!(Mantle.Plan(g))
    KA.synchronize(backend)
    @test all(==(7f0), Array(out))
end

@testset "a Ref argument is read per run, not per plan" begin
    # How everything that changes between samples reaches a recorded dispatch:
    # the sample index, the camera, the scene re-adapted every call. A plan
    # resolves its arguments once, so a value packed at compile time would be
    # the one it was built with for ever.
    backend = Lava.LavaBackend()
    dev = Hikari.mantle_device(backend)
    n = 64
    src = KA.allocate(backend, Float32, n)
    KA.fill!(src, 1f0)
    dst = Mantle.Buffer(dev, zeros(Float32, n))
    scale = Ref(3f0)

    g = Mantle.Graph(dev)
    Mantle.compute!(g, "scale") do p
        Mantle.dispatch!(p, Hikari.mantle_probe_scale!,
                         (Mantle.use(p, dst; write = true),
                          Mantle.use(p, src; read = true), scale), n)
    end
    plan = Mantle.Plan(g)
    Mantle.run!(plan)
    KA.synchronize(backend)
    @test all(==(3f0), Array(Mantle.storage(dst)))

    scale[] = 10f0
    Mantle.run!(plan)
    KA.synchronize(backend)
    @test all(==(10f0), Array(Mantle.storage(dst)))
end
