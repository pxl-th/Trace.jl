import{_ as a,o as n,c as e,aA as p}from"./chunks/framework.YyZVULjz.js";const h=JSON.parse('{"title":"","description":"","frontmatter":{},"headers":[],"relativePath":"get_started.md","filePath":"get_started.md","lastUpdated":null}'),i={name:"get_started.md"};function t(l,s,r,c,o,m){return n(),e("div",null,[...s[0]||(s[0]=[p(`<h2 id="Get-Started" tabindex="-1">Get Started <a class="header-anchor" href="#Get-Started" aria-label="Permalink to &quot;Get Started {#Get-Started}&quot;">​</a></h2><p>Hikari is a physically-based ray tracer for Julia. This guide shows you how to create your first rendered scene.</p><h3 id="Basic-Scene-Setup" tabindex="-1">Basic Scene Setup <a class="header-anchor" href="#Basic-Scene-Setup" aria-label="Permalink to &quot;Basic Scene Setup {#Basic-Scene-Setup}&quot;">​</a></h3><p>A minimal Hikari scene requires:</p><ol><li><p><strong>Geometry</strong> - Meshes to render (spheres, boxes, triangles)</p></li><li><p><strong>Materials</strong> - Surface properties (matte, mirror, glass)</p></li><li><p><strong>Lights</strong> - Light sources to illuminate the scene</p></li><li><p><strong>Camera</strong> - Viewpoint and image settings</p></li></ol><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>using GeometryBasics</span></span>
<span class="line"><span>using Hikari</span></span>
<span class="line"><span>using ImageShow</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Helper to tessellate GeometryBasics primitives into triangle meshes</span></span>
<span class="line"><span>to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Create materials using keyword constructors</span></span>
<span class="line"><span>red_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.2f0))</span></span>
<span class="line"><span>white_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.9f0))</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Build the scene using the push! API</span></span>
<span class="line"><span>scene = Hikari.Scene()</span></span>
<span class="line"><span>push!(scene, to_mesh(Sphere(Point3f(0, 0.5, 0), 0.5f0)), red_material)</span></span>
<span class="line"><span>push!(scene, to_mesh(Rect3f(Vec3f(-3, 0, -3), Vec3f(6, 0.01, 6))), white_material)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Add lights</span></span>
<span class="line"><span>push!(scene, Hikari.PointLight(Point3f(2f0, 3f0, -2f0), Hikari.RGBSpectrum(15f0)))</span></span>
<span class="line"><span>push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.1f0)))</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Finalize the acceleration structure</span></span>
<span class="line"><span>Hikari.sync!(scene)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Set up camera and film</span></span>
<span class="line"><span>resolution = Point2f(512, 512)</span></span>
<span class="line"><span>film = Hikari.Film(resolution)</span></span>
<span class="line"><span>camera = Hikari.PerspectiveCamera(</span></span>
<span class="line"><span>    Point3f(2f0, 2f0, -2f0), Point3f(0f0, 0.3f0, 0f0), film; fov=45f0,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>Hikari.clear!(film)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Render with the VolPath integrator</span></span>
<span class="line"><span>integrator = Hikari.VolPath(samples=16, max_depth=5)</span></span>
<span class="line"><span>integrator(scene, film, camera)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Postprocess and display</span></span>
<span class="line"><span>img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)</span></span>
<span class="line"><span>Array(img)</span></span></code></pre></div>`,6)])])}const u=a(i,[["render",t]]);export{h as __pageData,u as default};
