import{_ as a,o as n,c as e,aA as p}from"./chunks/framework.YyZVULjz.js";const h=JSON.parse('{"title":"","description":"","frontmatter":{},"headers":[],"relativePath":"materials.md","filePath":"materials.md","lastUpdated":null}'),l={name:"materials.md"};function i(t,s,r,c,o,f){return n(),e("div",null,[...s[0]||(s[0]=[p(`<h2 id="Materials-Showcase" tabindex="-1">Materials Showcase <a class="header-anchor" href="#Materials-Showcase" aria-label="Permalink to &quot;Materials Showcase {#Materials-Showcase}&quot;">​</a></h2><p>This example demonstrates the different material types available in Hikari, arranged in a scene to clearly show their properties.</p><h3 id="Available-Materials" tabindex="-1">Available Materials <a class="header-anchor" href="#Available-Materials" aria-label="Permalink to &quot;Available Materials {#Available-Materials}&quot;">​</a></h3><p>Hikari supports several physically-based materials:</p><ul><li><p><strong>MatteMaterial</strong>: Diffuse surfaces with optional roughness (Oren-Nayar model)</p></li><li><p><strong>MirrorMaterial</strong>: Perfect specular reflection</p></li><li><p><strong>GlassMaterial</strong>: Transparent material with refraction and optional roughness</p></li><li><p><strong>PlasticMaterial</strong>: Combination of diffuse and glossy specular reflection (coated diffuse)</p></li><li><p><strong>ConductorMaterial</strong>: Physically-based metals with complex IOR (eta + k)</p></li></ul><div class="language-@example vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">@example</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>using GeometryBasics</span></span>
<span class="line"><span>using Hikari</span></span>
<span class="line"><span>using FileIO</span></span>
<span class="line"><span>using ImageShow</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Helper to tessellate primitives</span></span>
<span class="line"><span>to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Helper to create a sphere resting on the ground</span></span>
<span class="line"><span>LowSphere(radius, x, z) = Sphere(Point3f(x, radius, z), radius)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># ============================================</span></span>
<span class="line"><span># Define all material types</span></span>
<span class="line"><span># ============================================</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 1. MATTE MATERIALS - Diffuse reflection</span></span>
<span class="line"><span># Smooth diffuse (Lambertian) - terracotta color</span></span>
<span class="line"><span>matte_smooth = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.4f0, 0.3f0))</span></span>
<span class="line"><span># Rough diffuse (Oren-Nayar) - chalk/clay appearance</span></span>
<span class="line"><span>matte_rough = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.85f0, 0.85f0, 0.8f0), σ=60f0)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 2. MIRROR - Perfect specular reflection</span></span>
<span class="line"><span># Silver mirror</span></span>
<span class="line"><span>mirror_silver = Hikari.MirrorMaterial(Kr=Hikari.RGBSpectrum(0.95f0, 0.93f0, 0.88f0))</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 3. CONDUCTOR MATERIALS - Physically-based metals</span></span>
<span class="line"><span># Gold</span></span>
<span class="line"><span>gold = Hikari.ConductorMaterial(</span></span>
<span class="line"><span>    eta=Hikari.RGBSpectrum(0.15557f0, 0.42415f0, 1.3831f0),</span></span>
<span class="line"><span>    k=Hikari.RGBSpectrum(3.6024f0, 2.4721f0, 1.9155f0),</span></span>
<span class="line"><span>    roughness=0.05f0,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span># Copper</span></span>
<span class="line"><span>copper = Hikari.ConductorMaterial(</span></span>
<span class="line"><span>    eta=Hikari.RGBSpectrum(0.27105f0, 0.67693f0, 1.3164f0),</span></span>
<span class="line"><span>    k=Hikari.RGBSpectrum(3.6092f0, 2.6248f0, 2.2921f0),</span></span>
<span class="line"><span>    roughness=0.1f0,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 4. GLASS MATERIALS - Refraction and transparency</span></span>
<span class="line"><span># Clear glass</span></span>
<span class="line"><span>glass_clear = Hikari.GlassMaterial(index=1.5f0)</span></span>
<span class="line"><span># Frosted glass</span></span>
<span class="line"><span>glass_frosted = Hikari.GlassMaterial(roughness=0.15f0, index=1.5f0)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># 5. PLASTIC MATERIALS - Diffuse + glossy specular</span></span>
<span class="line"><span># Shiny red plastic</span></span>
<span class="line"><span>plastic_shiny = Hikari.PlasticMaterial(</span></span>
<span class="line"><span>    Kd=Hikari.RGBSpectrum(0.8f0, 0.1f0, 0.1f0),</span></span>
<span class="line"><span>    Ks=Hikari.RGBSpectrum(0.4f0),</span></span>
<span class="line"><span>    roughness=0.02f0,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span># Matte blue plastic</span></span>
<span class="line"><span>plastic_matte = Hikari.PlasticMaterial(</span></span>
<span class="line"><span>    Kd=Hikari.RGBSpectrum(0.15f0, 0.3f0, 0.7f0),</span></span>
<span class="line"><span>    Ks=Hikari.RGBSpectrum(0.15f0),</span></span>
<span class="line"><span>    roughness=0.25f0,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Floor</span></span>
<span class="line"><span>floor_mat = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.85f0))</span></span>
<span class="line"><span></span></span>
<span class="line"><span># ============================================</span></span>
<span class="line"><span># Create scene with 3x3 grid of spheres</span></span>
<span class="line"><span># ============================================</span></span>
<span class="line"><span># Back row:   Matte Smooth | Matte Rough  | Glass Clear</span></span>
<span class="line"><span># Middle row: Mirror Silver| Gold         | Copper</span></span>
<span class="line"><span># Front row:  Frosted Glass| Plastic Shiny| Plastic Matte</span></span>
<span class="line"><span></span></span>
<span class="line"><span>r = 0.38f0</span></span>
<span class="line"><span>sx, sz = 1.0f0, 1.1f0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>scene = Hikari.Scene()</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Back row (z = sz)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, -sx, sz)), matte_smooth)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, 0f0, sz)), matte_rough)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, sx, sz)), glass_clear)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Middle row (z = 0)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, -sx, 0f0)), mirror_silver)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, 0f0, 0f0)), gold)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, sx, 0f0)), copper)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Front row (z = -sz)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, -sx, -sz)), glass_frosted)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, 0f0, -sz)), plastic_shiny)</span></span>
<span class="line"><span>push!(scene, to_mesh(LowSphere(r, sx, -sz)), plastic_matte)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Ground plane</span></span>
<span class="line"><span>push!(scene, to_mesh(Rect3f(Vec3f(-3, 0, -3), Vec3f(6, 0.01, 6))), floor_mat)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Lighting</span></span>
<span class="line"><span>push!(scene, Hikari.PointLight(Point3f(3f0, 4f0, -3f0), Hikari.RGBSpectrum(15f0)))</span></span>
<span class="line"><span>push!(scene, Hikari.PointLight(Point3f(-3f0, 3f0, -2f0), Hikari.RGBSpectrum(6f0)))</span></span>
<span class="line"><span>push!(scene, Hikari.PointLight(Point3f(0f0, 3f0, 3f0), Hikari.RGBSpectrum(8f0)))</span></span>
<span class="line"><span>push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.05f0)))</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Hikari.sync!(scene)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Camera setup</span></span>
<span class="line"><span>resolution = Point2f(1024)</span></span>
<span class="line"><span>film = Hikari.Film(resolution)</span></span>
<span class="line"><span>camera = Hikari.PerspectiveCamera(</span></span>
<span class="line"><span>    Point3f(0f0, 3.5f0, -5f0), Point3f(0f0, 0.2f0, 0f0), film; fov=40f0,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span>Hikari.clear!(film)</span></span>
<span class="line"><span></span></span>
<span class="line"><span># Render</span></span>
<span class="line"><span>integrator = Hikari.VolPath(samples=16, max_depth=5)</span></span>
<span class="line"><span>integrator(scene, film, camera)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)</span></span>
<span class="line"><span>Array(img)</span></span></code></pre></div><h3 id="Material-Properties-Reference" tabindex="-1">Material Properties Reference <a class="header-anchor" href="#Material-Properties-Reference" aria-label="Permalink to &quot;Material Properties Reference {#Material-Properties-Reference}&quot;">​</a></h3><table tabindex="0"><thead><tr><th style="text-align:right;">Material</th><th style="text-align:right;">Key Parameters</th><th style="text-align:right;">Best For</th></tr></thead><tbody><tr><td style="text-align:right;"><strong>MatteMaterial</strong></td><td style="text-align:right;"><code>Kd</code> (color), <code>σ</code> (roughness 0-90°)</td><td style="text-align:right;">Diffuse surfaces: walls, cloth, paper, chalk</td></tr><tr><td style="text-align:right;"><strong>MirrorMaterial</strong></td><td style="text-align:right;"><code>Kr</code> (reflectance color)</td><td style="text-align:right;">Perfect specular mirrors</td></tr><tr><td style="text-align:right;"><strong>GlassMaterial</strong></td><td style="text-align:right;"><code>Kr</code>, <code>Kt</code>, <code>roughness</code>, <code>index</code> (IOR)</td><td style="text-align:right;">Glass, water, gems, ice, frosted surfaces</td></tr><tr><td style="text-align:right;"><strong>PlasticMaterial</strong></td><td style="text-align:right;"><code>Kd</code>, <code>Ks</code>, <code>roughness</code></td><td style="text-align:right;">Plastic, painted surfaces, ceramics</td></tr><tr><td style="text-align:right;"><strong>ConductorMaterial</strong></td><td style="text-align:right;"><code>eta</code>, <code>k</code>, <code>roughness</code></td><td style="text-align:right;">Physically-based metals: gold, copper, silver</td></tr></tbody></table><h3 id="Index-of-Refraction-IOR-Reference" tabindex="-1">Index of Refraction (IOR) Reference <a class="header-anchor" href="#Index-of-Refraction-IOR-Reference" aria-label="Permalink to &quot;Index of Refraction (IOR) Reference {#Index-of-Refraction-IOR-Reference}&quot;">​</a></h3><p>Common IOR values for GlassMaterial:</p><ul><li><p>Air: 1.0</p></li><li><p>Water: 1.33</p></li><li><p>Glass: 1.5</p></li><li><p>Crystal: 1.6-2.0</p></li><li><p>Diamond: 2.42</p></li></ul>`,11)])])}const g=a(l,[["render",i]]);export{h as __pageData,g as default};
