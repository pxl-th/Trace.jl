"""
Hikari does not call `KernelAbstractions.synchronize`. It waits through Mantle.

**Why this is a source scan and not a behavioural test.** The two spellings do
the same thing to the GPU, so no render distinguishes them; what differs is who
knows the wait happened. `KA.synchronize(::LavaBackend)` flushes the batch queue
directly, so the stall is invisible to the layer that owns submission — Mantle
cannot see it, cannot avoid it, and cannot replace it with a device-side
predicate. That is not a property of any one call; it is a property of the
spelling, and a scan is the only thing that pins a spelling.

It also waits for more than the caller wanted: the dispatch queue AND the upload
queue, where the caller usually meant "the thing I just submitted".

What replaced them:

  * `Mantle.waitfor!(plan)` — wait for one plan's last run, submitting it if the
    host is still holding it. The bounce loop's early exit, which reads a
    device-written ray count between chunks and was the expensive one: measured
    at 10% of a baked materials render.
  * `Mantle.waitidle(device)` — the device, for the six setup-time launches that
    are not graph runs (aux buffers, majorant grids, light powers, postprocess).

`sync!(scene.accel)` is not in scope: `Raycore.sync!` is another package's, and
the line in `scene.jl` naming it is a comment.

Comments are excluded, and deliberately: several of them explain what a call
used to be, and deleting that history to satisfy a grep would be the wrong
trade.
"""

using Test

const HIKARI_SRC = normpath(joinpath(@__DIR__, "..", "src"))

"""Every `.jl` under `src/`, minus the generated trace."""
function hikari_sources()
    out = String[]
    for (root, _, files) in walkdir(HIKARI_SRC), f in files
        endswith(f, ".jl") || continue
        f == "precompile_statements.jl" && continue
        push!(out, joinpath(root, f))
    end
    sort!(out)
end

"""Strip line comments so the scan sees code, not prose.

Crude on purpose: a `#` inside a string literal would be treated as a comment,
which can only ever HIDE a match. Since the assertion is "no matches", a false
negative here would be a hole — so the pattern below is also required to match
a call, `synchronize(`, rather than the bare word.
"""
decomment(line) = (i = findfirst('#', line); i === nothing ? line : line[1:prevind(line, i)])

const SYNC_CALL = r"\b(KA|KernelAbstractions)\s*\.\s*synchronize\s*\("

@testset "Hikari waits through Mantle, never KernelAbstractions" begin
    offenders = Tuple{String,Int,String}[]
    for path in hikari_sources(), (i, line) in enumerate(eachline(path))
        occursin(SYNC_CALL, decomment(line)) || continue
        push!(offenders, (relpath(path, HIKARI_SRC), i, strip(line)))
    end
    if !isempty(offenders)
        for (f, i, l) in offenders
            @info "KA.synchronize in Hikari source" file = f line = i code = l
        end
    end
    @test isempty(offenders)

    # The scan can see a call at all — otherwise "no matches" would also be the
    # answer for a broken pattern, and this file would pass forever.
    @test occursin(SYNC_CALL, "    KA.synchronize(backend)")
    @test occursin(SYNC_CALL, "KernelAbstractions.synchronize(be)")
    @test !occursin(SYNC_CALL, decomment("# used to be KA.synchronize(backend)"))
end
