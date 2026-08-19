using Test
using Hikari, Lava, Mantle
import KernelAbstractions as KA

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
