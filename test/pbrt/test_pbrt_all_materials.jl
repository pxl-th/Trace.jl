using Test
using Hikari
using Lava
using GeometryBasics
using LinearAlgebra
using FileIO
using Statistics
using Colors
import KernelAbstractions as KA

const SCENES_DIR = joinpath(@__DIR__, "scenes")
const REFS_DIR = joinpath(@__DIR__, "references")
const PBRT_BIN = "/sim/Programmieren/VulkanDev/pbrt-v4/build/pbrt"

# Force lavapipe
ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
const BACKEND = Lava.LavaBackend()

# ============================================================================
# Image comparison — tile-based, following Makie ReferenceTests
# ============================================================================

"""
    compare_images(a, b; tile_size=30) -> Float64

Compare two images using tile-based max-of-mean color distance,
following Makie's ReferenceTests approach.

Divides the image into ~`tile_size`px tiles, computes the mean Euclidean
color distance within each tile, and returns the **maximum** across all tiles.

This catches spatial errors (flips, shifts, wrong highlights) that global
metrics like total energy ratio would miss entirely.
"""
function compare_images(a::AbstractMatrix, b::AbstractMatrix; tile_size=30)
    size(a) != size(b) && return Inf

    h, w = size(a)
    range_h = round.(Int, range(0, h, length=max(2, ceil(Int, h / tile_size))))
    range_w = round.(Int, range(0, w, length=max(2, ceil(Int, w / tile_size))))

    boundary_iter(boundaries) = zip(boundaries[1:end-1] .+ 1, boundaries[2:end])

    function color_dist(p1, p2)
        r1, g1, b1 = Float64(red(p1)), Float64(green(p1)), Float64(blue(p1))
        r2, g2, b2 = Float64(red(p2)), Float64(green(p2)), Float64(blue(p2))
        return sqrt((r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2)
    end

    return maximum(Iterators.product(boundary_iter(range_h), boundary_iter(range_w))) do ((r1, r2), (c1, c2))
        tile_a = @view a[r1:r2, c1:c2]
        tile_b = @view b[r1:r2, c1:c2]
        mean(color_dist.(tile_a, tile_b))
    end
end

"""
    compute_energy_ratio(a, b) -> Float64

Total energy ratio (sum of all pixel luminances).
"""
function compute_energy_ratio(a::AbstractMatrix, b::AbstractMatrix)
    total_ref = 0.0
    total_hik = 0.0
    for i in eachindex(a)
        rp = a[i]; hp = b[i]
        total_ref += Float64(red(rp)) + Float64(green(rp)) + Float64(blue(rp))
        total_hik += Float64(red(hp)) + Float64(green(hp)) + Float64(blue(hp))
    end
    return total_hik / max(total_ref, 1e-10)
end

# ============================================================================
# Reference generation and rendering
# ============================================================================

function ensure_reference(scene_name; spp=256)
    ref_file = joinpath(REFS_DIR, "$(scene_name).exr")
    if !isfile(ref_file)
        scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
        @info "Generating reference: $scene_name"
        run(`$PBRT_BIN $scene_file --outfile $ref_file --spp $spp`)
    end
    return FileIO.load(ref_file)
end

function render_and_compare(scene_name; hikari_spp=256, ref_spp=256)
    ref = ensure_reference(scene_name; spp=ref_spp)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    r = Hikari.load_pbrt(scene_file; backend=BACKEND, samples=hikari_spp)
    s = r.integrator_settings
    vp = Hikari.VolPath(samples=s.samples, max_depth=s.max_depth, regularize=s.regularize,
                        russian_roulette_depth=s.russian_roulette_depth,
                        max_component_value=s.max_component_value)
    vp(r.scene, r.film, r.camera)
    # Raw linear HDR framebuffer — matches pbrt's Film "rgb" EXR output
    # (no tonemapping, no gamma; sensor imagingRatio=1.0 for default cie1931)
    fb = Array(r.film.framebuffer)

    score = compare_images(ref, fb)
    energy = compute_energy_ratio(ref, fb)
    return (score=score, energy=energy)
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

# Tile-based score threshold:
#   Most correct 256spp renders: 0.01–0.07
#   Horizontally flipped render: 0.15–0.54
#   Genuine spatial bugs:        0.10+
const SCORE_THRESHOLD = 0.07
const ENERGY_LOW  = 0.95
const ENERGY_HIGH = 1.05

@testset "All Materials × All Lights" begin
    for mat in MATERIALS
        @testset "$mat" begin
            for light in LIGHTS
                scene_name = "mat_$(mat)_light_$(light)"
                scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
                isfile(scene_file) || continue

                @testset "$light" begin
                    m = render_and_compare(scene_name)
                    println("  $mat + $light: score=$(round(m.score, digits=4)) energy=$(round(m.energy, digits=3))")

                    @test m.score < SCORE_THRESHOLD
                    @test m.energy > ENERGY_LOW
                    @test m.energy < ENERGY_HIGH
                end
            end
        end
    end

    @testset "mix (diffuse+conductor)" begin
        scene_name = "mat_mix_light_point"
        scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
        if isfile(scene_file)
            m = render_and_compare(scene_name)
            println("  mix + point: score=$(round(m.score, digits=4)) energy=$(round(m.energy, digits=3))")
            @test m.score < SCORE_THRESHOLD
            @test m.energy > ENERGY_LOW
            @test m.energy < ENERGY_HIGH
        end
    end
end
