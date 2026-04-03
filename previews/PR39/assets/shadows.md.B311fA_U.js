import{_ as a,o as n,c as e,aA as p}from"./chunks/framework.YyZVULjz.js";const h=JSON.parse('{"title":"","description":"","frontmatter":{},"headers":[],"relativePath":"shadows.md","filePath":"shadows.md","lastUpdated":null}'),i={name:"shadows.md"};function l(t,s,r,c,o,m){return n(),e("div",null,[...s[0]||(s[0]=[p(`<h2 id="Shadows" tabindex="-1">Shadows <a class="header-anchor" href="#Shadows" aria-label="Permalink to &quot;Shadows {#Shadows}&quot;">​</a></h2><p>This example demonstrates a Cornell box-style scene with multiple spheres, different materials (matte, mirror, conductor), and shadow rendering.</p><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>using GeometryBasics</span></span>
<span class="line"><span>using Hikari</span></span>
<span class="line"><span>using ImageShow</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Helper to tessellate primitives</span></span>
<span class="line"><span>to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Define materials</span></span>
<span class="line"><span>material_white = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.9f0))</span></span>
<span class="line"><span>material_red = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.2f0))</span></span>
<span class="line"><span>material_green = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.2f0, 0.8f0, 0.2f0))</span></span>
<span class="line"><span>material_blue = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.3f0, 0.5f0, 0.9f0))</span></span>
<span class="line"><span>mirror = Hikari.MirrorMaterial(Kr=Hikari.RGBSpectrum(0.95f0))</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Build scene</span></span>
<span class="line"><span>scene = Hikari.Scene()</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Spheres with different materials</span></span>
<span class="line"><span>push!(scene, to_mesh(Sphere(Point3f(0, 0.5, 0), 0.5f0)), mirror)        # Center mirror</span></span>
<span class="line"><span>push!(scene, to_mesh(Sphere(Point3f(0.8, 0.3, 0.3), 0.3f0)), material_blue)</span></span>
<span class="line"><span>push!(scene, to_mesh(Sphere(Point3f(-0.8, 0.3, 0.3), 0.3f0)), material_red)</span></span>
<span class="line"><span>push!(scene, to_mesh(Sphere(Point3f(0, 0.25, 0.9), 0.25f0)), material_white)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Room geometry (Cornell box style, Y-up)</span></span>
<span class="line"><span>push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(4, 0.01, 4))), material_white)   # Floor</span></span>
<span class="line"><span>push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, 2 - 0.01), Vec3f(4, 3, 0.01))), material_white)  # Back wall</span></span>
<span class="line"><span>push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(0.01, 3, 4))), material_red)     # Left wall</span></span>
<span class="line"><span>push!(scene, to_mesh(Rect3f(Vec3f(2 - 0.01, 0, -2), Vec3f(0.01, 3, 4))), material_green)  # Right wall</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Lights</span></span>
<span class="line"><span>push!(scene, Hikari.PointLight(Point3f(0f0, 2.5f0, 0f0), Hikari.RGBSpectrum(12f0)))</span></span>
<span class="line"><span>push!(scene, Hikari.PointLight(Point3f(-1f0, 1.5f0, -1.5f0), Hikari.RGBSpectrum(4f0)))</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Hikari.sync!(scene)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Camera and film</span></span>
<span class="line"><span>resolution = Point2f(512, 512)</span></span>
<span class="line"><span>film = Hikari.Film(resolution)</span></span>
<span class="line"><span>camera = Hikari.PerspectiveCamera(</span></span>
<span class="line"><span>    Point3f(0f0, 1.5f0, -3f0), Point3f(0f0, 0.4f0, 0f0), film; fov=50f0,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>Hikari.clear!(film)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Render</span></span>
<span class="line"><span>integrator = Hikari.VolPath(samples=16, max_depth=10)</span></span>
<span class="line"><span>integrator(scene, film, camera)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)</span></span>
<span class="line"><span>Array(img)</span></span></code></pre></div>`,3)])])}const d=a(i,[["render",l]]);export{h as __pageData,d as default};
