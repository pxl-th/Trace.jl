# Would Mantle-owned work queues alias, and by how much?
#
# The one thing the Mantle port did not move is memory. Hikari's work queues are
# `KA.allocate` buffers the passes read as FOREIGN — a graph can order accesses
# to a buffer it does not own, but it cannot adopt one into its arena, and
# aliasing is the arena's. That is ~1.9 GiB of queues at 1280x1080 the placer
# never sees.
#
# Whether moving them is worth the work is a question about liveness, and the
# placer answers it exactly. So this models the bounce loop's pass structure with
# transients of the real footprints and asks — rather than reasoning about which
# queues overlap, which is how a first version of this got the wrong answer twice.
#
# THE ANSWER, on this structure, at 1280x1080:
#
#   rounds=1   peak 1392.2 MiB   naive 1566.2 MiB   saved 174.0 MiB
#   rounds=2   peak 1566.2 MiB   naive 1566.2 MiB   saved     0.0 MiB
#   rounds=4   ...                                  saved     0.0 MiB
#   rounds=8   ...                                  saved     0.0 MiB
#
# One round has exactly one disjoint pair — the shadow queue is born at medium
# direct lighting, by which time the medium-sample queue is dead — and 174.0 MiB
# is the shadow queue entire. Two rounds and it is gone, because every queue is
# refilled every round, so each is live from the first round to the last and
# nothing is disjoint any more.
#
# The shipped shape is a chunk of `EXIT_CHECK_INTERVAL` = 8 rounds. So moving
# ownership buys **nothing in aliasing** as the loop is written. It would still
# buy pool accounting — one allocator seeing every workload, which is what Mantle
# is for — but that is bookkeeping, not bytes, and it is not a reason on its own.
#
# What WOULD change the answer is a bounce loop where a queue is filled once and
# drained once rather than reused every round. That is a different algorithm, not
# a different declaration.
#
# Run: julia --project=. benchmarks/queue_aliasing/aliasing.jl

using Mantle
import KernelAbstractions as KA
const M = Mantle

const PX = 1280 * 1080

"Item sizes, as `sizeof` reports them for the work-item structs."
const SIZES = (ray = 164, medium_sample = 320, medium_scatter = 116,
               hit_surface = 304, shadow = 132, escaped = 128,
               hit_area_light = 172, typed_ref = 4)

"""
`nrounds` bounces, with every intermediate queue a transient of its real size.

The ray queues and the film are deliberately absent: they cross plan boundaries,
so they are persistent whatever else changes, and including them would only add a
constant to both columns.

The counter reset is modelled as it is written — one pass writing the 4-byte size
counters and NOT the payloads. That distinction is the whole experiment: declared
as touching the whole queue it would pin every payload live from the first pass
and the saving would be zero even for one round.
"""
function chunk_probe(dev, nrounds; n_material_types = 4)
    g = M.Graph(dev)
    T(bytes) = M.Transient.Buffer(g, UInt8, bytes * PX)
    q = (medium_sample = T(SIZES.medium_sample),
         medium_scatter = T(SIZES.medium_scatter),
         hit_surface = T(SIZES.hit_surface),
         shadow = T(SIZES.shadow),
         escaped = T(SIZES.escaped),
         hit_area_light = T(SIZES.hit_area_light),
         per_material = [T(SIZES.typed_ref) for _ in 1:n_material_types])
    counters = [M.Transient.Buffer(g, UInt8, 4) for _ in 1:(6 + n_material_types)]

    for r in 1:nrounds
        M.compute!(g, "reset$r") do p
            for c in counters; M.use(p, c; write = true); end
        end
        M.compute!(g, "trace$r") do p
            for b in (q.escaped, q.medium_sample, q.hit_surface,
                      q.hit_area_light, q.per_material...)
                M.use(p, b; read = true, write = true)
            end
        end
        M.compute!(g, "medium-sample$r") do p
            M.use(p, q.medium_sample; read = true)
            for b in (q.medium_scatter, q.hit_surface, q.hit_area_light,
                      q.escaped, q.per_material...)
                M.use(p, b; read = true, write = true)
            end
        end
        M.compute!(g, "medium-dl$r") do p
            M.use(p, q.medium_scatter; read = true)
            M.use(p, q.shadow; read = true, write = true)
        end
        M.compute!(g, "medium-scatter$r") do p
            M.use(p, q.medium_scatter; read = true)
        end
        M.compute!(g, "escaped$r") do p; M.use(p, q.escaped; read = true); end
        M.compute!(g, "emitters$r") do p; M.use(p, q.hit_area_light; read = true); end
        M.compute!(g, "shade$r") do p
            M.use(p, q.hit_surface; read = true)
            for b in q.per_material; M.use(p, b; read = true); end
        end
        M.compute!(g, "shadow$r") do p; M.use(p, q.shadow; read = true); end
    end
    return g
end

mib(x) = round(x / 2^20; digits = 1)

function report(dev = M.Device(M.Host()), rounds = (1, 2, 4, 8))
    println("work queues as transients, $(PX) px:")
    for nr in rounds
        pl = M.Plan(chunk_probe(dev, nr))
        println("  rounds=", rpad(nr, 3),
                " peak ",  lpad(mib(M.peakbytes(pl)), 7), " MiB",
                "   naive ", lpad(mib(pl.naive), 7), " MiB",
                "   saved ", lpad(mib(pl.naive - M.peakbytes(pl)), 7), " MiB")
        M.free!(pl)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    report()
end
