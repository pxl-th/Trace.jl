"""
What Hikari still names from Lava, and why each one is allowed to stay.

This file used to be a debt ledger. `Lava.` in Hikari's source meant a renderer
reaching into a Vulkan runtime, and the list existed to stop it growing while the
runtime was moved out. On 2026-08-27 it was: **22 names across 3 files.**

The move happened. Lava is a Julia→SPIR-V compiler with no Vulkan dependency, and
everything Hikari used to reach for — `LavaBackend`, `VulkanTLAS`, `vk_context`,
`RayTracingPipeline`, `@compile_workload` — is Mantle's. What is left is 13
names in ONE file, and they are a different kind of thing entirely:

    lava_rt_launch_id_x   lava_rt_trace_ray        lava_rt_primitive_id
    lava_rt_ray_tmax      lava_rt_instance_id      lava_rt_instance_custom_index
    lava_rt_hit_bary_u    lava_rt_hit_bary_v       lava_rt_reorder_thread
    lava_rt_hit_object_trace_ray   lava_rt_hit_object_execute_shader
    lava_rt_payload_store_f32_at   lava_rt_payload_load_f32_at

These are **device-side ray-tracing intrinsics** — what a shader BODY calls,
lowered by the compiler to `OpTraceRayKHR` and friends. They are the ray-tracing
counterpart of what `KernelInterface` holds for compute (`get_global_id`,
`shfl_down`, `barrier`), and they are named from Lava for the same reason a
compute kernel names KI: that is where shader vocabulary lives.

So the ledger's meaning inverted. It no longer says "this should be zero"; it
says **this is the shader vocabulary, and nothing else may join it**. A
`Lava.LavaArray` appearing here would mean the split had come undone. They go to
KI if it ever grows a ray-tracing half, and this list is the exact worklist for
that.

`src/precompile_statements.jl` is skipped: generated from a compile trace, it
names whatever types the run saw.

Parsed rather than grepped — `import Lava: a, b,` continues across lines, and a
regex either misses the continuation or matches the word in a comment.
"""

using Test

# One file, one kind of name. Written out rather than pattern-matched on the
# `lava_rt_` prefix, so that adding an intrinsic is a deliberate line here.
const ALLOWED_LAVA_REFS = Dict(
    "src/integrators/volpath/rt-pipeline.jl" => Dict(
        "lava_rt_launch_id_x" => 1,
        "lava_rt_trace_ray" => 1,
        "lava_rt_hit_object_trace_ray" => 1,
        "lava_rt_reorder_thread" => 1,
        "lava_rt_hit_object_execute_shader" => 1,
        "lava_rt_payload_store_f32_at" => 1,
        "lava_rt_payload_load_f32_at" => 1,
        "lava_rt_primitive_id" => 1,
        "lava_rt_instance_id" => 1,
        "lava_rt_instance_custom_index" => 1,
        "lava_rt_ray_tmax" => 1,
        "lava_rt_hit_bary_u" => 1,
        "lava_rt_hit_bary_v" => 1,
    ),
)

const GENERATED_SOURCES = ["precompile_statements.jl"]

"""
Every `Lava.<name>` and every name in an `import Lava: …` list, as `name =>
count`.

`Lava` alone — `import Lava`, and the qualifier of a counted `Lava.foo` — is not
a reference to anything. What matters is which of Lava's names Hikari knows.
"""
function lava_references(path::AbstractString)
    found = Dict{String, Int}()
    note!(name) = (found[name] = get(found, name, 0) + 1)

    function walk(ex)
        ex isa Expr || return
        if ex.head === :. && length(ex.args) == 2 &&
                ex.args[1] === :Lava && ex.args[2] isa QuoteNode
            note!(String(ex.args[2].value))
            return
        end
        if (ex.head === :import || ex.head === :using) && length(ex.args) == 1 &&
                ex.args[1] isa Expr && ex.args[1].head === :(:)
            spec = ex.args[1]
            if spec.args[1] isa Expr && spec.args[1].head === :. &&
                    spec.args[1].args == [:Lava]
                for name in spec.args[2:end]
                    name isa Expr && name.head === :. && note!(String(name.args[1]))
                end
                return
            end
        end
        foreach(walk, ex.args)
        return
    end

    walk(Meta.parseall(read(path, String)))
    return found
end

@testset "Hikari names only shader vocabulary from Lava" begin
    src = joinpath(pkgdir(Hikari), "src")

    actual = Dict{String, Dict{String, Int}}()
    for (root, _, files) in walkdir(src), file in files
        endswith(file, ".jl") || continue
        file in GENERATED_SOURCES && continue
        path = joinpath(root, file)
        refs = lava_references(path)
        isempty(refs) || (actual[relpath(path, pkgdir(Hikari))] = refs)
    end

    # A second file naming Lava is the case this catches. After the move there is
    # exactly one, and it is the ray-tracing shader.
    @test sort(collect(keys(actual))) == sort(collect(keys(ALLOWED_LAVA_REFS)))

    for (file, allowed) in ALLOWED_LAVA_REFS
        @testset "$file" begin
            refs = get(actual, file, Dict{String, Int}())
            @test isempty(setdiff(keys(refs), keys(allowed)))
            gone = setdiff(keys(allowed), keys(refs))
            isempty(gone) ||
                @info "$file no longer references $(join(sort(collect(gone)), ", ")) — delete these from ALLOWED_LAVA_REFS"
            @test isempty(gone)
            for (name, n) in allowed
                @test get(refs, name, 0) == n
            end
        end
    end

    # Every one is a device-side intrinsic. This is the claim the docstring makes
    # and the reason the list is allowed to be non-empty: a name here that is not
    # shader vocabulary is a runtime reference that crept back in.
    @testset "and nothing but shader vocabulary" begin
        for (_, allowed) in ALLOWED_LAVA_REFS, name in keys(allowed)
            @test startswith(name, "lava_rt_")
        end
    end
end
