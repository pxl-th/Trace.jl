#
# pbrt reference suite — drives Hikari through every shipped .pbrt scene
# and compares against the committed pbrt-v4 EXR references.
#
# Defines `run_pbrt_suite(; backend, samples, hw_accel, ...)`; the bottom of
# this file auto-runs with sensible defaults when invoked as a standalone
# script (via `julia test_pbrt_all_materials.jl`) but NOT when included from
# runtests.jl (runtests picks the parameters and invokes the function
# explicitly).

using Test
using Hikari
using Lava

include(joinpath(@__DIR__, "suite.jl"))

# ── Defaults ────────────────────────────────────────────────────────────────

const DEFAULT_SPP = 256

# Tile p95 (log-space): correct renders < 0.05, spatial bugs > 0.10
const TILE_THRESHOLD = 0.07
const ENERGY_LOW     = 0.90
const ENERGY_HIGH    = 1.10

const BASE_MATERIALS = [
    "diffuse", "diffuse_colored",
    "conductor_gold", "conductor_mirror", "conductor_rough",
    "conductor_silver", "conductor_copper",
    "dielectric", "dielectric_rough", "dielectric_rough_high", "dielectric_diamond",
    "thindielectric",
    "coateddiffuse", "coateddiffuse_rough",
    "coatedconductor", "coatedconductor_rough",
    "diffusetransmission",
]
const BASE_LIGHTS = ["point", "distant", "spot", "area", "ambient"]

# ── Test helpers (parametrized) ─────────────────────────────────────────────

function _test_scene(scene_name; backend, samples, hw_accel,
                     tile_thresh=TILE_THRESHOLD,
                     energy_low=ENERGY_LOW, energy_high=ENERGY_HIGH,
                     ref_spp=samples)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    isfile(scene_file) || return nothing

    ref = ensure_reference(scene_name; spp=ref_spp)
    fb  = render_scene(scene_name; backend, samples, hw_accel)
    m   = compute_metrics(ref, fb)

    println("  $(scene_name): tile=$(round(m.tile, digits=4)) energy=$(round(m.energy_ratio, digits=3))")
    @test m.tile         < tile_thresh
    @test m.energy_ratio > energy_low
    @test m.energy_ratio < energy_high
    return m
end

function _test_scenes_matching(prefix; backend, samples, hw_accel, kwargs...)
    for name in list_scenes(prefix)
        @testset "$name" begin
            _test_scene(name; backend, samples, hw_accel, kwargs...)
        end
    end
end

# ── Main entry point ────────────────────────────────────────────────────────

"""
    run_pbrt_suite(; backend=Lava.LavaBackend(), samples=DEFAULT_SPP,
                     hw_accel=false, media_energy=(0.80, 1.20))

Run the full pbrt reference comparison under `@testset`s. All parameters are
explicit — callers (runtests.jl, standalone invocation) pick the backend and
hw_accel flag. Skips scenes whose `.pbrt` is missing.
"""
function run_pbrt_suite(; backend=Lava.LavaBackend(),
                          samples::Int=DEFAULT_SPP,
                          hw_accel::Bool=false,
                          media_energy::Tuple{Float64,Float64}=(0.80, 1.20))
    label = hw_accel ? "HW RT" : "SW BVH"
    @testset "pbrt reference ($label, $samples spp)" begin
        @testset "Materials × Lights" begin
            for mat in BASE_MATERIALS
                @testset "$mat" begin
                    for light in BASE_LIGHTS
                        scene = "mat_$(mat)_light_$(light)"
                        @testset "$light" begin
                            _test_scene(scene; backend, samples, hw_accel)
                        end
                    end
                end
            end
            @testset "mix" begin
                _test_scene("mat_mix_light_point"; backend, samples, hw_accel)
            end
        end

        @testset "Textures"       _test_scenes_matching("tex_";    backend, samples, hw_accel)
        @testset "Light variants" _test_scenes_matching("light_";  backend, samples, hw_accel)
        @testset "Filters"        _test_scenes_matching("filter_"; backend, samples, hw_accel)
        @testset "Sensors"        _test_scenes_matching("sensor_"; backend, samples, hw_accel)
        @testset "Media" begin
            _test_scenes_matching("medium_"; backend, samples, hw_accel,
                                  energy_low=media_energy[1], energy_high=media_energy[2])
        end
    end
end

# ── Standalone invocation ───────────────────────────────────────────────────
# When this file is the entry point (e.g. `julia test_pbrt_all_materials.jl`)
# run the suite in SW mode by default, and also in HW mode if the device
# supports it. When included from runtests.jl we do nothing here —
# runtests.jl calls `run_pbrt_suite` explicitly with its own parameters.

if abspath(PROGRAM_FILE) == @__FILE__
    run_pbrt_suite(; hw_accel=false)
    if hw_rt_available()
        run_pbrt_suite(; hw_accel=true)
    end
end
