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

# Force lavapipe
ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
const BACKEND = Lava.LavaBackend()

function ensure_reference(scene_name; spp=256)
    ref_file = joinpath(REFS_DIR, "$(scene_name).exr")
    if !isfile(ref_file)
        scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
        @info "Generating reference: $scene_name"
        run(`$PBRT_BIN $scene_file --outfile $ref_file --spp $spp`)
    end
    return FileIO.load(ref_file)
end

function render_and_compare(scene_name; hikari_spp=64, max_depth=8, ref_spp=256)
    ref = ensure_reference(scene_name; spp=ref_spp)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    r = Hikari.load_pbrt(scene_file; backend=BACKEND, samples=hikari_spp, max_depth=max_depth)
    vp = Hikari.VolPath(samples=hikari_spp, max_depth=max_depth)
    vp(r.scene, r.film, r.camera)
    Hikari.postprocess!(r.film)
    fb = r.film.framebuffer

    h, w = size(ref)
    total_ref = 0.0
    total_hik = 0.0
    diffs = Float32[]
    for j in 1:w, i in 1:h
        rp = ref[i, j]; hp = fb[i, j]
        rr, rg, rb = Float32(rp.r), Float32(rp.g), Float32(rp.b)
        hr, hg, hb = Float32(hp.r), Float32(hp.g), Float32(hp.b)
        total_ref += rr + rg + rb
        total_hik += hr + hg + hb
        push!(diffs, (abs(rr-hr) + abs(rg-hg) + abs(rb-hb)) / 3)
    end

    energy_ratio = total_hik / max(total_ref, 1e-10)
    return (energy=energy_ratio, mean=mean(diffs), median=median(diffs))
end

# ============================================================================
# All material × light combinations
# ============================================================================

const MATERIALS = [
    "diffuse", "conductor_gold", "conductor_mirror",
    "dielectric", "dielectric_rough", "thindielectric",
    "coateddiffuse", "coatedconductor", "diffusetransmission",
]

const LIGHTS = ["point", "distant", "spot", "area", "ambient"]

@testset "All Materials × All Lights" begin
    for mat in MATERIALS
        @testset "$mat" begin
            for light in LIGHTS
                scene_name = "mat_$(mat)_light_$(light)"
                scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
                isfile(scene_file) || continue

                @testset "$light" begin
                    m = render_and_compare(scene_name)
                    println("  $mat + $light: energy=$(round(m.energy, digits=3)) mean=$(round(m.mean, digits=4))")

                    # Energy conservation: within 10% (generous for 64 vs 256 spp)
                    @test m.energy > 0.85
                    @test m.energy < 1.15
                    # Mean diff reasonable
                    @test m.mean < 0.16
                end
            end
        end
    end

    @testset "mix (diffuse+conductor)" begin
        scene_name = "mat_mix_light_point"
        scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
        if isfile(scene_file)
            m = render_and_compare(scene_name)
            println("  mix + point: energy=$(round(m.energy, digits=3)) mean=$(round(m.mean, digits=4))")
            @test m.energy > 0.85
            @test m.energy < 1.15
            @test m.mean < 0.16
        end
    end
end
