# Emissive - Area light emission data
# Used inside MediumInterface.arealight to add emission to any surface

# ============================================================================
# Emissive Type
# ============================================================================

"""
    Emissive{LeTex}

Emission data for area lights. Always used inside `MediumInterface.arealight`
to add light emission to surfaces. A surface with an arealight both reflects
light (via the MediumInterface's BSDF material) AND emits light.

# Fields
* `Le`: Emitted radiance (color/intensity texture or TextureRef)
* `scale`: Intensity multiplier applied to Le
* `two_sided`: If true, emits from both sides of the surface

# Usage
```julia
# Area light panel (matching pbrt-v4's "rgb L" [3 3 3]):
Emissive(Le=(3, 3, 3), scale=1.0, two_sided=true)
```
"""
struct Emissive{LeTex} <: Material
    Le::LeTex  # Texture, Raycore.TextureRef, or raw RGBSpectrum
    scale::Float32
    two_sided::Bool
end

# ============================================================================
# User-friendly keyword constructor
# ============================================================================

"""
    Emissive(; Le=RGBSpectrum(1), scale=1.0, two_sided=false)

Create an emissive material for area lights. Photometric normalization is applied
automatically (matching pbrt-v4). `Le` values correspond directly to pbrt's `"rgb L"`.

# Examples
```julia
# Equivalent to pbrt-v4: AreaLightSource "diffuse" "rgb L" [3 3 3]
Emissive(Le=(3, 3, 3), scale=1.0, two_sided=true)
```
"""
function Emissive(;
    Le=RGBSpectrum(1f0),
    scale::Real=1f0,
    two_sided::Bool=false
)
    # Apply photometric normalization matching pbrt-v4's area light creation:
    # scale /= SpectrumToPhotometric(Le_spectrum)
    normalized_scale = Float32(scale) / D65_PHOTOMETRIC
    Emissive(to_texture(Le), normalized_scale, two_sided)
end

# ============================================================================
# Material Interface Implementation
# ============================================================================

"""
    get_emission(mat::Emissive, si::SurfaceInteraction) -> RGBSpectrum

Get the emitted radiance at a surface point.
Returns zero if the surface is one-sided and we're on the back.
"""
@propagate_inbounds function get_emission(mat::Emissive, wo::Vec3f, n::Vec3f, uv::Point2f)
    # Check if we're on the emitting side
    cos_theta = dot(wo, n)
    if !mat.two_sided && cos_theta < 0f0
        return RGBSpectrum(0f0)
    end
    # Evaluate Le texture at UV
    Le = mat.Le(uv)
    return Le * mat.scale
end

"""
    get_emission(mat::Emissive, uv::Point2f) -> RGBSpectrum

Get the emitted radiance at UV coordinates (without directional check).
"""
@propagate_inbounds function get_emission(mat::Emissive, uv::Point2f)
    Le = mat.Le(uv)
    return Le * mat.scale
end

# Base fallbacks for non-emissive materials are in material.jl (included before this file)

"""
    is_emissive(mat::Material) -> Bool

Check if a material emits light.
"""
@propagate_inbounds is_emissive(::Emissive) = true

"""
    is_pure_emissive(mat::Material) -> Bool

Check if a material is purely emissive (no BSDF, only emits light).
"""
@propagate_inbounds is_pure_emissive(::Emissive) = true

# ============================================================================
# GPU Support
# ============================================================================

"""
    to_gpu(ArrayType, mat::Emissive)

Convert Emissive to GPU-compatible form.
"""
function to_gpu(ArrayType, mat::Emissive)
    Le_gpu = to_gpu(ArrayType, mat.Le)
    return Emissive(Le_gpu, mat.scale, mat.two_sided)
end

# ============================================================================
# Albedo extraction for denoising auxiliary buffers
# ============================================================================

"""
    get_albedo(mat::Emissive, uv::Point2f) -> RGBSpectrum

Get the "albedo" of an emissive material for denoising.
For emissive materials, we return the normalized emission color.
"""
@propagate_inbounds function get_albedo(mat::Emissive, uv::Point2f)
    Le = mat.Le(uv)
    # Return normalized color (so it's in 0-1 range for denoising)
    luminance = to_Y(Le)
    if luminance > 0f0
        return Le / luminance
    else
        return RGBSpectrum(0f0)
    end
end

