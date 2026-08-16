using Test
using Hikari
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using JET
using Lava

include("materials.jl")
include("type_stability.jl")
include("film.jl")
include("gpu_compat.jl")
include("volpath_integration.jl")
include("denoise.jl")
include("test_caching_gc_correctness.jl")
include("test_texture_wrap.jl")
include("test_checkerboard_texture.jl")
include("test_hw_sw_parity.jl")
include("test_update_material_null.jl")
include("test_multitypeset_updates.jl")
include("test_material_type_collapse.jl")
include("test_volpath_per_iter_lifecycle.jl")

# ── pbrt reference suite ────────────────────────────────────────────────────
# Runs the full pbrt-v4 reference image comparison against every scene in
# test/pbrt/scenes for which a committed reference EXR exists.  Driven by
# `run_pbrt_suite(; backend, samples, hw_accel)` (defined in
# test/pbrt/test_pbrt_all_materials.jl).
#
# SW BVH always runs. HW RT runs additionally when the bound Vulkan device
# reports `rt_pipeline_properties`. Both are opt-out via env vars so CI can
# disable paths that are known to be broken on a given runner without
# editing this file.
include(joinpath(@__DIR__, "pbrt", "test_pbrt_all_materials.jl"))

const PBRT_SPP = parse(Int, get(ENV, "HIKARI_PBRT_SPP", "256"))

if get(ENV, "HIKARI_SKIP_PBRT_SW", "false") != "true"
    run_pbrt_suite(; samples=PBRT_SPP, hw_accel=false)
end

if get(ENV, "HIKARI_SKIP_PBRT_HW", "false") != "true" && hw_rt_available()
    run_pbrt_suite(; samples=PBRT_SPP, hw_accel=true)
end
