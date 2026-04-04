using Test
using Hikari
using Lava
using GeometryBasics
using LinearAlgebra
using FileIO
using Statistics
using Colors
using ImageFiltering
import KernelAbstractions as KA

const SCENES_DIR = joinpath(@__DIR__, "scenes")
const REFS_DIR = joinpath(@__DIR__, "references")
const PBRT_BIN = "/sim/Programmieren/VulkanDev/pbrt-v4/build/pbrt"

# Force lavapipe
ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
const BACKEND = Lava.LavaBackend()

const SPP = 256

# ============================================================================
# Image comparison — log-space tile p95 + energy ratio
# ============================================================================
#
# Log-space tile comparison is robust to MC noise because:
#  1. log1p compresses HDR highlights, reducing noise amplitude in bright areas.
#  2. 32px tiles average out per-pixel variance.
#  3. 95th percentile ignores a few noisy tiles (MC noise is localized).
# This catches spatial bugs (flips, shifts, wrong highlights) that affect
# many tiles simultaneously, while tolerating per-tile MC variance.

"""
    downsample_2x(img) -> Matrix

Average 2x2 blocks to halve resolution. Reduces MC noise by 2x.
"""
function downsample_2x(img::AbstractMatrix)
    blurred = imfilter(img, Kernel.gaussian((0.75, 0.75)))
    return blurred[1:2:end, 1:2:end]
end

"""
    tile_score(a, b; tile_size=16, percentile=0.95) -> Float64

95th-percentile tile-based mean color distance in log1p space.
Images are downsampled 2x before comparison to reduce MC noise.
"""
function tile_score(a::AbstractMatrix, b::AbstractMatrix; tile_size=16, percentile=0.95)
    size(a) != size(b) && return Inf

    # Downsample 2x to average out MC noise
    a = downsample_2x(a)
    b = downsample_2x(b)

    h, w = size(a)
    range_h = round.(Int, range(0, h, length=max(2, ceil(Int, h / tile_size))))
    range_w = round.(Int, range(0, w, length=max(2, ceil(Int, w / tile_size))))
    boundary_iter(boundaries) = zip(boundaries[1:end-1] .+ 1, boundaries[2:end])

    function log_color_dist(p1, p2)
        r1 = log1p(max(0.0, Float64(red(p1)))); g1 = log1p(max(0.0, Float64(green(p1)))); b1 = log1p(max(0.0, Float64(blue(p1))))
        r2 = log1p(max(0.0, Float64(red(p2)))); g2 = log1p(max(0.0, Float64(green(p2)))); b2 = log1p(max(0.0, Float64(blue(p2))))
        return sqrt((r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2)
    end

    scores = [
        mean(log_color_dist.(@view(a[r1:r2, c1:c2]), @view(b[r1:r2, c1:c2])))
        for (r1, r2) in boundary_iter(range_h)
        for (c1, c2) in boundary_iter(range_w)
    ]

    sort!(scores)
    idx = clamp(ceil(Int, percentile * length(scores)), 1, length(scores))
    return scores[idx]
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

function ensure_reference(scene_name; spp=SPP)
    ref_file = joinpath(REFS_DIR, "$(scene_name).exr")
    if !isfile(ref_file)
        scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
        @info "Generating reference: $scene_name"
        run(`$PBRT_BIN $scene_file --outfile $ref_file --spp $spp`)
    end
    return FileIO.load(ref_file)
end

function render_and_compare(scene_name; hikari_spp=SPP, ref_spp=SPP)
    ref = ensure_reference(scene_name; spp=ref_spp)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    r = Hikari.load_pbrt(scene_file; backend=BACKEND, samples=hikari_spp)
    s = r.integrator_settings
    vp = Hikari.VolPath(samples=hikari_spp, max_depth=s.max_depth, regularize=s.regularize,
                        russian_roulette_depth=s.russian_roulette_depth,
                        max_component_value=s.max_component_value,
                        sensor=r.sensor, sensor_name=r.sensor_name)
    vp(r.scene, r.film, r.camera)
    fb = Array(r.film.framebuffer)

    tile = tile_score(ref, fb)
    energy = compute_energy_ratio(ref, fb)
    return (tile=tile, energy=energy)
end

# ============================================================================
# Thresholds
# ============================================================================

# Tile p95 (log-space): correct renders < 0.05, spatial bugs > 0.10
# Energy: correct 0.90–1.10 (wider for media/sensor tests)
const TILE_THRESHOLD = 0.07
const ENERGY_LOW     = 0.90
const ENERGY_HIGH    = 1.10

# ============================================================================
# Test helpers
# ============================================================================

function test_scene(scene_name; tile_thresh=TILE_THRESHOLD,
                    energy_low=ENERGY_LOW, energy_high=ENERGY_HIGH)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    isfile(scene_file) || return nothing
    m = render_and_compare(scene_name)
    println("  $(scene_name): tile=$(round(m.tile, digits=4)) energy=$(round(m.energy, digits=3))")
    @test m.tile   < tile_thresh
    @test m.energy > energy_low
    @test m.energy < energy_high
    return m
end

function test_scenes_matching(prefix; kwargs...)
    scene_files = sort(filter(f -> startswith(f, prefix) && endswith(f, ".pbrt"), readdir(SCENES_DIR)))
    for fname in scene_files
        name = replace(fname, ".pbrt" => "")
        @testset "$name" begin
            test_scene(name; kwargs...)
        end
    end
end

# ============================================================================
# Material × Light tests
# ============================================================================

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

@testset "Materials × Lights" begin
    for mat in BASE_MATERIALS
        @testset "$mat" begin
            for light in BASE_LIGHTS
                scene_name = "mat_$(mat)_light_$(light)"
                @testset "$light" begin
                    test_scene(scene_name)
                end
            end
        end
    end

    @testset "mix" begin
        test_scene("mat_mix_light_point")
    end
end

# ============================================================================
# Texture tests
# ============================================================================

@testset "Textures" begin
    test_scenes_matching("tex_")
end

# ============================================================================
# Light variant tests
# ============================================================================

@testset "Light variants" begin
    test_scenes_matching("light_")
end

# ============================================================================
# Filter tests
# ============================================================================

@testset "Filters" begin
    test_scenes_matching("filter_")
end

# ============================================================================
# Sensor tests
# ============================================================================

@testset "Sensors" begin
    test_scenes_matching("sensor_")
end

# ============================================================================
# Medium tests (wider energy tolerance for volumetric scattering)
# ============================================================================

@testset "Media" begin
    test_scenes_matching("medium_"; energy_low=0.80, energy_high=1.20)
end
