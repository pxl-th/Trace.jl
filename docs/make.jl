using Documenter
using DocumenterVitepress
using Hikari
using FileIO, Colors, ImageFiltering, Statistics

# ============================================================================
# Generate pbrt comparison page
# ============================================================================

const DOCS_DIR     = @__DIR__
const ASSETS_CMP   = joinpath(DOCS_DIR, "src", "public", "pbrt-comparison")
const TEST_DIR     = joinpath(DOCS_DIR, "..", "test", "pbrt")
const DISPLAY_DIR  = joinpath(TEST_DIR, "display")
const REFS_DIR     = joinpath(TEST_DIR, "references")
const RECORDED_DIR = joinpath(TEST_DIR, "recorded")
const SCENES_DIR   = joinpath(TEST_DIR, "scenes")

mkpath(ASSETS_CMP)

function tile_score_exr(ref_img, rec_img; tile_size=16, percentile=0.95)
    size(ref_img) == size(rec_img) || return Inf
    a = imfilter(ref_img, Kernel.gaussian((0.75, 0.75)))[1:2:end, 1:2:end]
    b = imfilter(rec_img, Kernel.gaussian((0.75, 0.75)))[1:2:end, 1:2:end]
    h, w = size(a)
    rh = round.(Int, range(0, h, length=max(2, ceil(Int, h / tile_size))))
    rw = round.(Int, range(0, w, length=max(2, ceil(Int, w / tile_size))))
    bnd(r) = zip(r[1:end-1] .+ 1, r[2:end])
    dist(p1, p2) = begin
        r1 = log1p(max(0.0, Float64(red(p1)))); g1 = log1p(max(0.0, Float64(green(p1)))); b1 = log1p(max(0.0, Float64(blue(p1))))
        r2 = log1p(max(0.0, Float64(red(p2)))); g2 = log1p(max(0.0, Float64(green(p2)))); b2 = log1p(max(0.0, Float64(blue(p2))))
        sqrt((r1-r2)^2+(g1-g2)^2+(b1-b2)^2)
    end
    scores = [mean(dist.(a[r1:r2,c1:c2], b[r1:r2,c1:c2]))
              for (r1,r2) in bnd(rh) for (c1,c2) in bnd(rw)]
    sort!(scores)
    return scores[clamp(ceil(Int, percentile * length(scores)), 1, length(scores))]
end

function generate_comparison_page()
    scene_files = isdir(SCENES_DIR) ? sort(filter(f -> endswith(f, ".pbrt"), readdir(SCENES_DIR))) : String[]

    struct_data = NamedTuple{(:name, :tile, :energy), Tuple{String, Float64, Float64}}[]

    for fname in scene_files
        name = replace(fname, ".pbrt" => "")
        ref_png = joinpath(DISPLAY_DIR, "ref_$(name).png")
        rec_png = joinpath(DISPLAY_DIR, "rec_$(name).png")
        (isfile(ref_png) && isfile(rec_png)) || continue

        cp(ref_png, joinpath(ASSETS_CMP, "ref_$(name).png"); force=true)
        cp(rec_png, joinpath(ASSETS_CMP, "rec_$(name).png"); force=true)

        tile = 0.0; energy = 1.0
        ref_exr = joinpath(REFS_DIR, "$(name).exr")
        rec_exr = joinpath(RECORDED_DIR, "$(name).exr")
        if isfile(ref_exr) && isfile(rec_exr)
            try
                ref_img = FileIO.load(ref_exr)
                rec_img = FileIO.load(rec_exr)
                total_ref = sum(Float64(red(p)) + Float64(green(p)) + Float64(blue(p)) for p in ref_img)
                total_rec = sum(Float64(red(p)) + Float64(green(p)) + Float64(blue(p)) for p in rec_img)
                energy = total_rec / max(total_ref, 1e-10)
                tile   = tile_score_exr(ref_img, rec_img)
            catch e
                @warn "Could not compute score for $name: $e"
            end
        end
        push!(struct_data, (name=name, tile=tile, energy=energy))
    end

    sort!(struct_data; by=s -> (-abs(s.energy - 1.0), -s.tile))

    n = length(struct_data)
    n_energy_ok = Base.count(s -> 0.95 < s.energy < 1.05, struct_data)
    n_tile_ok   = Base.count(s -> s.tile < 0.07, struct_data)
    println("  Comparison: $n scenes — energy ok: $n_energy_ok/$n — tile<0.07: $n_tile_ok/$n")

    # Build JSON array for embedding in the standalone gallery HTML
    json_entries = map(struct_data) do s
        """{"name":"$(s.name)","tile":$(round(s.tile, digits=4)),"energy":$(round(s.energy, digits=4)),"ref":"ref_$(s.name).png","rec":"rec_$(s.name).png"}"""
    end
    json_array = "[\n" * join(json_entries, ",\n") * "\n]"

    # Write standalone gallery HTML (iframe target — avoids Vue SSR entity-encoding of JS)
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

    open(joinpath(ASSETS_CMP, "gallery.html"), "w") do io
        print(io, gallery_html)
    end

    # Write comparison.md — iframe embeds the standalone gallery
    # (avoids Vue SSR entity-encoding JavaScript string literals)
    page_content = """## Hikari vs pbrt-v4 Comparison

Hikari is validated against [pbrt-v4](https://github.com/mmp/pbrt-v4) across $(n) test scenes covering all material types, light types, and rendering configurations.
Each scene is rendered at 256 samples per pixel. Click any thumbnail to toggle between Hikari and the pbrt-v4 reference.

The **tile score** is a perceptual difference metric (log-space L2 over 16x16 tiles, p95). The **energy ratio** is total scene luminance Hikari / pbrt-v4; values outside 0.95–1.05 are highlighted in red.

```@raw html
<iframe src="/pbrt-comparison/gallery.html"
  style="width:100%; height:900px; border:none; border-radius:8px;"
  title="Hikari vs pbrt-v4 comparison gallery">
</iframe>
```
"""

    open(joinpath(DOCS_DIR, "src", "comparison.md"), "w") do io
        print(io, page_content)
    end

    if n == 0
        @warn "No pbrt comparison images found — comparison page will show empty gallery"
    end
end

println("Generating pbrt comparison page...")
generate_comparison_page()

# ============================================================================
# Build docs
# ============================================================================

makedocs(; sitename = "Hikari", authors = "Anton Smirnov, Simon Danisch and contributors",
    modules = [Hikari],
    checkdocs = :exports,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/JuliaGraphics/Hikari.jl",
        devbranch = "master",
        devurl = "dev";
    ),
    draft = false,
    source = "src",
    build = "build",
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "Get Started" => "get_started.md",
        "Rendering" => "shadows.md",
        "Materials" => "materials.md",
        "Lights" => "lights.md",
        "pbrt-v4 Comparison" => "comparison.md",
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaGraphics/Hikari.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "master",
    push_preview = true,
)
