## Materials

Hikari implements the full set of physically-based materials from pbrt-v4. All material parameters accept either constant `RGBSpectrum`/`Float32` values or image textures.

```@setup materials
using GeometryBasics, Hikari, ImageShow
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)
sphere(r, x, z) = to_mesh(Sphere(Point3f(x, r, z), r))
floor_mat = Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.82f0))

function render_scene(materials; res=768, spp=32)
    r = 0.38f0; sx = 1.0f0; sz = 1.1f0
    scene = Hikari.Scene()
    positions = [(-sx, sz), (0f0, sz), (sx, sz),
                 (-sx, 0f0), (0f0, 0f0), (sx, 0f0),
                 (-sx, -sz), (0f0, -sz), (sx, -sz)]
    for (mat, (x, z)) in zip(materials, positions)
        push!(scene, sphere(r, x, z), mat)
    end
    push!(scene, to_mesh(Rect3f(Vec3f(-3, 0, -3), Vec3f(6, 0.01, 6))), floor_mat)
    push!(scene, Hikari.PointLight(Point3f(3f0, 4f0, -3f0), Hikari.RGBSpectrum(40f0)))
    push!(scene, Hikari.PointLight(Point3f(-3f0, 3f0, -2f0), Hikari.RGBSpectrum(20f0)))
    push!(scene, Hikari.PointLight(Point3f(0f0, 3f0, 3f0), Hikari.RGBSpectrum(25f0)))
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
    Hikari.sync!(scene)
    film   = Hikari.Film(Point2f(res, res))
    camera = Hikari.PerspectiveCamera(
        Point3f(0f0, 3.5f0, -5f0), Point3f(0f0, 0.2f0, 0f0), film; fov=40f0)
    Hikari.clear!(film)
    Hikari.VolPath(samples=spp, max_depth=8)(scene, film, camera)
    return Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.0f0, gamma=2.2f0))
end
```

### Diffuse

Lambertian diffuse reflection. Set `σ > 0` to switch to the Oren-Nayar model for rough surfaces like chalk or concrete.

```julia
Hikari.Diffuse(Kd=RGBSpectrum(0.8f0))                          # white Lambertian
Hikari.Diffuse(Kd=RGBSpectrum(0.8f0, 0.2f0, 0.1f0))           # red
Hikari.Diffuse(Kd=RGBSpectrum(0.9f0), σ=45f0)                  # rough diffuse (Oren-Nayar)
```

| Parameter | Description |
|-----------|-------------|
| `Kd` | Diffuse reflectance color |
| `σ` | Roughness in degrees: 0 = Lambertian, higher = Oren-Nayar |

### Mirror

Perfect specular reflection. `Kr` tints the reflected color — use white for a silver mirror, or a gold hue for colored mirrors.

```julia
Hikari.Mirror(Kr=RGBSpectrum(0.95f0))                           # silver mirror
Hikari.Mirror(Kr=RGBSpectrum(0.9f0, 0.75f0, 0.3f0))           # gold-tinted mirror
```

### Conductor (Metals)

Physically-based metals using the complex index of refraction (η, k). The `roughness` parameter controls the GGX microfacet spread.

```julia
Hikari.Conductor(
    eta=RGBSpectrum(0.156f0, 0.424f0, 1.383f0),  # real part of IOR
    k=RGBSpectrum(3.602f0, 2.472f0, 1.916f0),     # extinction coefficient
    roughness=0.05f0,
)
```

**Built-in presets** with measured spectral data:

```julia
Hikari.Gold(roughness=0.0f0)
Hikari.Silver(roughness=0.0f0)
Hikari.Copper(roughness=0.1f0)
```

| Parameter | Description |
|-----------|-------------|
| `eta` | Real part of complex IOR (RGB) |
| `k` | Imaginary part / extinction coefficient (RGB) |
| `roughness` | GGX roughness (0 = mirror-like, 1 = fully rough) |
| `reflectance` | Optional color tint (default white) |
| `remap_roughness` | Remap perceptual roughness to GGX α (default true) |

### Dielectric (Glass / Water / Diamond)

Smooth or rough transparent material. Fresnel equations determine the reflection/transmission split at each interface.

```julia
Hikari.Dielectric(index=1.5f0)                     # clear glass
Hikari.Dielectric(index=1.33f0)                    # water
Hikari.Dielectric(index=2.42f0)                    # diamond
Hikari.Dielectric(roughness=0.15f0, index=1.5f0)  # frosted glass
```

Common IOR values: air 1.0 · water 1.33 · glass 1.5 · crystal 1.8 · diamond 2.42

| Parameter | Description |
|-----------|-------------|
| `index` | Index of refraction |
| `roughness` | GGX roughness (anisotropic: pass tuple `(u, v)`) |
| `Kr` | Reflection tint (default white) |
| `Kt` | Transmission tint (default white) |

### ThinDielectric

Thin-film glass (window pane, microscope slide). Unlike `Dielectric`, refracted rays are not bent — they pass straight through, accounting for multiple internal reflections analytically.

```julia
Hikari.ThinDielectric(eta=1.5f0)
```

### Plastic (Coated Diffuse)

A dielectric coating (Fresnel, glossy) layered over a diffuse base. This matches the typical appearance of painted plastic, ceramics, and lacquered wood.

```julia
Hikari.Plastic(color=(0.8f0, 0.1f0, 0.1f0), roughness=0.02f0)  # shiny red
Hikari.Plastic(color=(0.2f0, 0.4f0, 0.8f0), roughness=0.3f0)   # matte blue
```

For full control use `CoatedDiffuse` directly:

```julia
Hikari.CoatedDiffuse(
    reflectance=RGBSpectrum(0.6f0, 0.1f0, 0.1f0),
    roughness=0.05f0,
    eta=1.5f0,
    thickness=0.01f0,
)
```

### DiffuseTransmission

A flat, diffuse material that scatters light into both the reflected and transmitted hemispheres. Useful for thin cloth, paper, or leaf surfaces.

```julia
Hikari.DiffuseTransmission(
    reflectance=RGBSpectrum(0.2f0, 0.4f0, 0.1f0),
    transmittance=RGBSpectrum(0.1f0, 0.3f0, 0.05f0),
)
```

### Materials Showcase

Nine spheres demonstrating the available materials:

```@example materials
mats = [
    Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.8f0, 0.4f0, 0.3f0)),           # terracotta
    Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.85f0), σ=60f0),                 # chalk
    Hikari.Dielectric(index=1.5f0),                                          # glass
    Hikari.Mirror(Kr=Hikari.RGBSpectrum(0.95f0, 0.93f0, 0.88f0)),          # silver mirror
    Hikari.Gold(roughness=0.02f0),                                           # gold
    Hikari.Copper(roughness=0.08f0),                                         # copper
    Hikari.Dielectric(roughness=0.12f0, index=1.5f0),                       # frosted glass
    Hikari.Plastic(color=(0.8f0, 0.1f0, 0.1f0), roughness=0.02f0),         # shiny plastic
    Hikari.DiffuseTransmission(                                               # paper
        reflectance=Hikari.RGBSpectrum(0.4f0),
        transmittance=Hikari.RGBSpectrum(0.4f0)),
]

render_scene(mats; spp=64)
```

Row 1 (back): Lambertian · Oren-Nayar · Dielectric
Row 2 (middle): Mirror · Gold · Copper
Row 3 (front): Frosted glass · Plastic · DiffuseTransmission
