#!/usr/bin/env julia
#
# Render all pbrt test scenes with Hikari, compare against pre-committed
# pbrt-v4 reference EXRs, and generate the comparison gallery for docs.
#
# Outputs (all under test/pbrt/):
#   recorded/<name>.exr      - Hikari-rendered EXRs
#   display/ref_<name>.png   - tonemapped pbrt reference
#   display/rec_<name>.png   - tonemapped Hikari output
#   display/gallery.html     - standalone comparison gallery
#   display/scores.csv       - name,tile,energy per scene
#
# Exit code 1 if any scene fails energy (5%) or tile (<0.07) thresholds.

using FileIO, Colors
using Hikari, Lava

include(joinpath(@__DIR__, "suite.jl"))

const RECORDED_DIR = joinpath(@__DIR__, "recorded")
const DISPLAY_DIR  = joinpath(@__DIR__, "display")
# 256 spp produces the cleanest gallery thumbnails locally; on CI
# (lavapipe, no HW accel) it eats the 120-minute job timeout across 142
# scenes, so we default to a lower spp there. Override with
# `HIKARI_PBRT_SPP=256` to get the full gallery on capable hardware.
const SPP          = parse(Int, get(ENV, "HIKARI_PBRT_SPP", "32"))
const HW_ACCEL     = get(ENV, "HIKARI_HW_ACCEL", "false") == "true"

mkpath(RECORDED_DIR)
mkpath(DISPLAY_DIR)

# ============================================================================
# Helpers
# ============================================================================

function tonemap_to_png(exr_path, png_path)
    isfile(png_path) && return
    img = FileIO.load(exr_path)
    out = similar(img, RGB{Float64})
    for i in eachindex(img)
        p = img[i]
        r = Float64(red(p)); g = Float64(green(p)); b = Float64(blue(p))
        r = r / (1.0 + r); g = g / (1.0 + g); b = b / (1.0 + b)
        gamma(x) = x <= 0.0031308 ? 12.92x : 1.055 * x^(1/2.4) - 0.055
        out[i] = RGB(gamma(clamp(r,0,1)), gamma(clamp(g,0,1)), gamma(clamp(b,0,1)))
    end
    FileIO.save(png_path, out)
end

# ============================================================================
# Step 1: Render missing Hikari scenes
# ============================================================================

function render_all_missing(scene_files; backend, samples, hw_accel,
                            scenes_dir, refs_dir, out_dir)
    n_rendered = 0
    for (i, fname) in enumerate(scene_files)
        name    = replace(fname, ".pbrt" => "")
        rec_exr = joinpath(out_dir, "$(name).exr")
        isfile(rec_exr) && continue
        ref_exr = joinpath(refs_dir, "$(name).exr")
        isfile(ref_exr) || continue
        try
            fb = render_scene(name; backend, samples, hw_accel)
            FileIO.save(rec_exr, fb)
            n_rendered += 1
            println("  [$i/$(length(scene_files))] $name")
        catch e
            println("  [$i/$(length(scene_files))] SKIP $name: $(sprint(showerror, e; context=:limit=>120))")
        end
    end
    return n_rendered
end

println("Step 1: Rendering Hikari scenes at $SPP spp (hw_accel=$HW_ACCEL)...")
scene_files = sort(filter(f -> endswith(f, ".pbrt"), readdir(SCENES_DIR)))
backend = Lava.LavaBackend()
n_rendered = render_all_missing(scene_files;
    backend=backend, samples=SPP, hw_accel=HW_ACCEL,
    scenes_dir=SCENES_DIR, refs_dir=REFS_DIR, out_dir=RECORDED_DIR)
n_rendered > 0 && println("  Rendered $n_rendered scenes")

# ============================================================================
# Step 2: Compute scores and generate display PNGs
# ============================================================================

println("Step 2: Computing scores and generating display PNGs...")

struct SceneResult
    name::String
    tile::Float64
    energy::Float64
end

results = SceneResult[]
for fname in scene_files
    name = replace(fname, ".pbrt" => "")
    ref_exr = joinpath(REFS_DIR, "$(name).exr")
    rec_exr = joinpath(RECORDED_DIR, "$(name).exr")
    (isfile(ref_exr) && isfile(rec_exr)) || continue

    tonemap_to_png(ref_exr, joinpath(DISPLAY_DIR, "ref_$(name).png"))
    tonemap_to_png(rec_exr, joinpath(DISPLAY_DIR, "rec_$(name).png"))

    ref_img = FileIO.load(ref_exr)
    rec_img = FileIO.load(rec_exr)
    push!(results, SceneResult(name,
        tile_score(ref_img, rec_img),
        energy_ratio(ref_img, rec_img)))
end

sort!(results; by=s -> (-abs(s.energy - 1.0), -s.tile))

n = length(results)
n_energy_ok = Base.count(s -> 0.95 < s.energy < 1.05, results)
n_tile_ok = Base.count(s -> s.tile < 0.07, results)
println("  $n scenes | energy within 5%: $n_energy_ok/$n | tile<0.07: $n_tile_ok/$n")

# Write scores CSV
open(joinpath(DISPLAY_DIR, "scores.csv"), "w") do io
    println(io, "name,tile,energy")
    for s in results
        println(io, "$(s.name),$(s.tile),$(s.energy)")
    end
end

# ============================================================================
# Step 3: Generate standalone gallery HTML
# ============================================================================

println("Step 3: Generating gallery...")
json_entries = map(results) do s
    """{"name":"$(s.name)","tile":$(round(s.tile, digits=4)),"energy":$(round(s.energy, digits=4)),"ref":"ref_$(s.name).png","rec":"rec_$(s.name).png"}"""
end
json_array = "[\n" * join(json_entries, ",\n") * "\n]"

gallery_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  body { margin:0; background:#111; font-family:monospace; }
  #header { color:#aaa; text-align:center; padding:12px 0 8px; font-size:13px; }
  #grid { display:flex; flex-wrap:wrap; justify-content:center; padding:0 8px 16px; }
  .card { border-radius:8px; padding:8px; margin:4px; background:#1a1a1a; width:220px; }
  .card-title { margin:4px 0; font-size:13px; color:white; font-weight:bold; word-break:break-all; }
  .card-scores { font-size:11px; margin-bottom:4px; }
  .card-img { position:relative; width:100%; padding-bottom:100%; overflow:hidden; cursor:pointer; }
  .card-img img { width:100%; position:absolute; top:0; left:0; image-rendering:pixelated; }
  .card-label { text-align:center; font-size:11px; color:#aaa; margin-top:4px; }
</style>
</head>
<body>
<div id="header">Loading...</div>
<div id="grid"></div>
<script>
var scenes = $(json_array);
var nEok = 0, nTok = 0;
for (var i = 0; i < scenes.length; i++) {
  if (scenes[i].energy > 0.95 && scenes[i].energy < 1.05) nEok++;
  if (scenes[i].tile < 0.07) nTok++;
}
document.getElementById('header').textContent =
  scenes.length + ' scenes | energy within 5%: ' + nEok + '/' + scenes.length +
  ' | tile<0.07: ' + nTok + '/' + scenes.length + ' | click to toggle';

var grid = document.getElementById('grid');
for (var i = 0; i < scenes.length; i++) {
  (function(s) {
    var eok = s.energy > 0.95 && s.energy < 1.05;
    var sc = !eok ? '#E74C3C' : s.tile > 0.10 ? '#F39C12' : s.tile > 0.03 ? '#FDD835' : '#4CAF50';
    var bc = (!eok || s.tile > 0.10) ? sc : '#333';
    var ec = eok ? '#aaa' : '#E74C3C';
    var short = s.name.replace('mat_', '').replace(/_light_/g, ' + ');

    var card = document.createElement('div');
    card.className = 'card';
    card.style.border = '2px solid ' + bc;

    var title = document.createElement('div');
    title.className = 'card-title';
    title.textContent = short;

    var scores = document.createElement('div');
    scores.className = 'card-scores';
    var tileSpan = document.createElement('span');
    tileSpan.style.color = sc;
    tileSpan.textContent = 'tile: ' + s.tile.toFixed(3);
    var sep = document.createElement('span');
    sep.style.color = '#555';
    sep.textContent = ' | ';
    var energySpan = document.createElement('span');
    energySpan.style.color = ec;
    energySpan.textContent = 'energy: ' + s.energy.toFixed(3);
    scores.appendChild(tileSpan);
    scores.appendChild(sep);
    scores.appendChild(energySpan);

    var wrap = document.createElement('div');
    wrap.className = 'card-img';
    var imgRec = document.createElement('img');
    imgRec.src = s.rec;
    imgRec.style.zIndex = '2';
    var imgRef = document.createElement('img');
    imgRef.src = s.ref;
    imgRef.style.zIndex = '1';
    wrap.appendChild(imgRec);
    wrap.appendChild(imgRef);

    var label = document.createElement('div');
    label.className = 'card-label';
    label.textContent = 'Hikari';

    wrap.addEventListener('click', function(rec, lbl) {
      return function() {
        if (lbl.textContent === 'Hikari') { rec.style.opacity = '0'; lbl.textContent = 'pbrt-v4'; }
        else { rec.style.opacity = '1'; lbl.textContent = 'Hikari'; }
      };
    }(imgRec, label));

    card.appendChild(title);
    card.appendChild(scores);
    card.appendChild(wrap);
    card.appendChild(label);
    grid.appendChild(card);
  })(scenes[i]);
}
</script>
</body>
</html>
"""

open(joinpath(DISPLAY_DIR, "gallery.html"), "w") do io
    print(io, gallery_html)
end
println("  Gallery written to $(joinpath(DISPLAY_DIR, "gallery.html"))")

# ============================================================================
# Step 4: Assert thresholds
# ============================================================================

bad_energy = filter(s -> !(0.95 < s.energy < 1.05), results)
bad_tile = filter(s -> s.tile >= 0.07, results)
for s in bad_energy
    println("  FAIL energy: $(s.name) energy=$(round(s.energy, digits=4))")
end
for s in bad_tile
    println("  FAIL tile: $(s.name) tile=$(round(s.tile, digits=4))")
end

if !isempty(bad_energy) || !isempty(bad_tile)
    error("pbrt comparison FAILED: $(length(bad_energy)) energy failures, $(length(bad_tile)) tile failures")
end

println("All $n scenes pass thresholds.")
