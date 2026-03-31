using Test
using Hikari
using Lava
using GeometryBasics
using LinearAlgebra
using FileIO
using Statistics
import KernelAbstractions as KA

const SCENES_DIR = joinpath(@__DIR__, "scenes")
const REFS_DIR = joinpath(@__DIR__, "references")
const PBRT_BIN = "/sim/Programmieren/VulkanDev/pbrt-v4/build/pbrt"

# Force lavapipe for deterministic software rendering
ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

backend = Lava.LavaBackend()

# ============================================================================
# Comparison utilities
# ============================================================================

function generate_pbrt_reference(scene_name; spp=1024)
    ref_file = joinpath(REFS_DIR, "$(scene_name).exr")
    if !isfile(ref_file)
        scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
        @info "Generating reference: $scene_name at $(spp)spp"
        run(`$PBRT_BIN $scene_file --outfile $ref_file --spp $spp`)
    end
    return FileIO.load(ref_file)
end

function render_hikari(scene_name; spp=256, max_depth=10)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    r = Hikari.load_pbrt(scene_file; backend=backend, samples=spp, max_depth=max_depth)
    vp = Hikari.VolPath(samples=spp, max_depth=max_depth)
    vp(r.scene, r.film, r.camera)
    Hikari.postprocess!(r.film)
    return r.film.framebuffer
end

function compute_image_metrics(ref_img, hikari_fb)
    h, w = size(ref_img)
    @assert size(hikari_fb) == (h, w)

    total_ref = 0.0
    total_hikari = 0.0
    diffs = Float32[]

    for j in 1:w, i in 1:h
        rp = ref_img[i, j]
        hp = hikari_fb[i, j]
        rr, rg, rb = Float32(rp.r), Float32(rp.g), Float32(rp.b)
        hr, hg, hb = Float32(hp.r), Float32(hp.g), Float32(hp.b)

        total_ref += rr + rg + rb
        total_hikari += hr + hg + hb
        push!(diffs, (abs(rr - hr) + abs(rg - hg) + abs(rb - hb)) / 3)
    end

    return (
        mean_diff = mean(diffs),
        median_diff = median(diffs),
        max_diff = maximum(diffs),
        energy_ratio = total_hikari / max(total_ref, 1e-10),
        pct_above_005 = 100 * sum(d -> d > 0.05, diffs) / length(diffs),
        pct_above_01 = 100 * sum(d -> d > 0.1, diffs) / length(diffs),
    )
end

function compare_scene(scene_name; ref_spp=1024, hikari_spp=256, max_depth=10,
                       max_mean_diff=0.06, min_energy_ratio=0.95, max_energy_ratio=1.05)
    ref = generate_pbrt_reference(scene_name; spp=ref_spp)
    fb = render_hikari(scene_name; spp=hikari_spp, max_depth=max_depth)
    m = compute_image_metrics(ref, fb)

    println("  Scene: $scene_name")
    println("    Mean diff: $(round(m.mean_diff, digits=4))")
    println("    Energy ratio: $(round(m.energy_ratio, digits=4))")
    println("    Pixels >0.05: $(round(m.pct_above_005, digits=1))%")
    println("    Pixels >0.1:  $(round(m.pct_above_01, digits=1))%")

    # Core invariants:
    # 1. Total energy must be conserved (within 5%)
    @test m.energy_ratio > min_energy_ratio
    @test m.energy_ratio < max_energy_ratio
    # 2. Mean per-pixel difference should be small
    # (note: noise at different spp means this won't be zero)
    @test m.mean_diff < max_mean_diff
    # 3. No catastrophic failures — median should be low even if fireflies exist
    @test m.median_diff < max_mean_diff

    return m
end

# ============================================================================
# Comparison tests
# ============================================================================
@testset "PBRT vs Hikari Rendering Comparison" begin
    @testset "Shared geometry — Diffuse + Point light" begin
        compare_scene("shared_geom_diffuse"; max_depth=5)
    end

    @testset "Single sphere — Point light" begin
        compare_scene("single_sphere_point"; max_depth=5)
    end

    @testset "Single sphere — Distant light" begin
        compare_scene("single_sphere_distant"; max_depth=5)
    end

    @testset "Single sphere — Area light" begin
        compare_scene("single_sphere_area"; max_depth=5)
    end

    @testset "All materials" begin
        compare_scene("all_materials"; max_depth=10, max_mean_diff=0.08)
    end

    @testset "Named materials" begin
        compare_scene("named_materials"; max_depth=5)
    end

    @testset "Transforms" begin
        compare_scene("transforms"; max_depth=5)
    end
end
