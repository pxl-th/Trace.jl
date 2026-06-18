#
# pbrt reference suite — shared helpers
#
# Single source of truth for:
#   * locating scenes + references
#   * rendering a scene with Hikari (VolPath), parametrized by backend / spp /
#     max_depth / hw_accel
#   * loading a pre-committed pbrt-v4 EXR reference (optionally generating it
#     if the pbrt binary is available)
#   * image comparison metrics (log-space tile p95, energy ratio, mean/median
#     per-pixel diff)
#
# This file defines functions only — it does not run tests. It is `include`d
# by test_pbrt_all_materials.jl (normal @testset runner) and by
# run_comparison.jl (gallery-generating script). Both drive the same
# render/compare code so they can't drift.

using Hikari
using Lava
using FileIO
using Statistics
using Colors
using ImageFiltering
import KernelAbstractions as KA

const SCENES_DIR = joinpath(@__DIR__, "scenes")
const REFS_DIR   = joinpath(@__DIR__, "references")
const PBRT_BIN   = get(ENV, "PBRT_BIN", "/sim/Programmieren/VulkanDev/pbrt-v4/build/pbrt")

"""Is hardware ray tracing available on the currently-bound Lava device?"""
function hw_rt_available()
    try
        Lava.vk_context().rt_pipeline_properties !== nothing
    catch
        false
    end
end

# ── Reference management ────────────────────────────────────────────────────

"""
    ensure_reference(scene_name; spp) -> Matrix{RGB{Float32}}

Load the pbrt-v4 reference EXR for `scene_name`. If missing, fall back to
running the pbrt binary at `PBRT_BIN` to generate it. Throws if neither is
available.
"""
function ensure_reference(scene_name::AbstractString; spp::Int=1024)
    ref_file = joinpath(REFS_DIR, "$(scene_name).exr")
    if !isfile(ref_file)
        if !isfile(PBRT_BIN)
            error("No reference at $ref_file and pbrt binary not found at $PBRT_BIN")
        end
        scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
        @info "Generating reference: $scene_name at $(spp)spp"
        run(`$PBRT_BIN $scene_file --outfile $ref_file --spp $spp`)
    end
    return FileIO.load(ref_file)
end

# ── Rendering ───────────────────────────────────────────────────────────────

"""
    render_scene(scene_name; backend, samples, max_depth, hw_accel, close_vp=true)
        -> Matrix{RGB{Float32}}

Render `scene_name` with Hikari's VolPath integrator and return a host-side
framebuffer. All knobs are explicit — no defaults baked into globals. When
`hw_accel=true` the scene is built with `HWTLAS` and VolPath dispatches to
the hardware RT path.
"""
function render_scene(scene_name::AbstractString;
                      backend=Lava.LavaBackend(),
                      samples::Int=256,
                      max_depth::Union{Nothing,Int}=nothing,
                      hw_accel::Bool=false,
                      close_vp::Bool=true)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    isfile(scene_file) || error("pbrt scene not found: $scene_file")

    r = Hikari.load_pbrt(scene_file;
                         backend=backend,
                         samples=samples,
                         max_depth=max_depth,
                         hw_accel=hw_accel)
    s = r.integrator_settings
    vp = Hikari.VolPath(samples=samples,
                        max_depth=s.max_depth,
                        regularize=s.regularize,
                        russian_roulette_depth=s.russian_roulette_depth,
                        max_component_value=s.max_component_value,
                        sensor=r.sensor,
                        hw_accel=hw_accel)
    try
        vp(r.scene, r.film, r.camera)
        return Array(r.film.framebuffer)
    finally
        close_vp && close(vp)
    end
end

# ── Metrics ─────────────────────────────────────────────────────────────────

"""2×2 gaussian-filtered downsample. Reduces MC noise by ~2× so tile scores
are dominated by spatial bugs rather than per-pixel variance."""
function downsample_2x(img::AbstractMatrix)
    blurred = imfilter(img, Kernel.gaussian((0.75, 0.75)))
    return blurred[1:2:end, 1:2:end]
end

"""
    tile_score(a, b; tile_size=16, percentile=0.95) -> Float64

95th-percentile tile-based mean color distance in log1p space. Images are
downsampled 2× first. Returns `Inf` if sizes disagree.
"""
function tile_score(a::AbstractMatrix, b::AbstractMatrix;
                    tile_size::Int=16, percentile::Float64=0.95)
    size(a) == size(b) || return Inf
    a = downsample_2x(a); b = downsample_2x(b)
    h, w = size(a)
    rh = round.(Int, range(0, h, length=max(2, ceil(Int, h / tile_size))))
    rw = round.(Int, range(0, w, length=max(2, ceil(Int, w / tile_size))))
    bnd(r) = zip(r[1:end-1] .+ 1, r[2:end])
    function d(p1, p2)
        r1 = log1p(max(0.0, Float64(red(p1))));   g1 = log1p(max(0.0, Float64(green(p1))));   b1 = log1p(max(0.0, Float64(blue(p1))))
        r2 = log1p(max(0.0, Float64(red(p2))));   g2 = log1p(max(0.0, Float64(green(p2))));   b2 = log1p(max(0.0, Float64(blue(p2))))
        sqrt((r1-r2)^2+(g1-g2)^2+(b1-b2)^2)
    end
    scores = [mean(d.(@view(a[r1:r2,c1:c2]), @view(b[r1:r2,c1:c2])))
              for (r1,r2) in bnd(rh) for (c1,c2) in bnd(rw)]
    sort!(scores)
    return scores[clamp(ceil(Int, percentile * length(scores)), 1, length(scores))]
end

"""Total-luminance ratio: `sum(rec) / sum(ref)`. 1.0 = matched energy."""
function energy_ratio(ref::AbstractMatrix, rec::AbstractMatrix)
    total_ref = 0.0; total_rec = 0.0
    @inbounds for i in eachindex(ref, rec)
        rp = ref[i]; hp = rec[i]
        total_ref += Float64(red(rp)) + Float64(green(rp)) + Float64(blue(rp))
        total_rec += Float64(red(hp)) + Float64(green(hp)) + Float64(blue(hp))
    end
    return total_rec / max(total_ref, 1e-10)
end

"""All metrics in one pass."""
function compute_metrics(ref::AbstractMatrix, rec::AbstractMatrix)
    @assert size(ref) == size(rec) "size mismatch: $(size(ref)) vs $(size(rec))"
    diffs = Float32[]
    sizehint!(diffs, length(ref))
    @inbounds for i in eachindex(ref, rec)
        rp = ref[i]; hp = rec[i]
        rr, rg, rb = Float32(red(rp)), Float32(green(rp)), Float32(blue(rp))
        hr, hg, hb = Float32(red(hp)), Float32(green(hp)), Float32(blue(hp))
        push!(diffs, (abs(rr-hr) + abs(rg-hg) + abs(rb-hb)) / 3)
    end
    return (
        tile         = tile_score(ref, rec),
        energy_ratio = energy_ratio(ref, rec),
        mean_diff    = mean(diffs),
        median_diff  = median(diffs),
        max_diff     = maximum(diffs),
    )
end

"""Scene names present on disk under `SCENES_DIR` (without `.pbrt` suffix),
filtered by `prefix`. Sorted for determinism."""
function list_scenes(prefix::AbstractString="")
    files = filter(f -> startswith(f, prefix) && endswith(f, ".pbrt"), readdir(SCENES_DIR))
    return sort!([replace(f, ".pbrt" => "") for f in files])
end
