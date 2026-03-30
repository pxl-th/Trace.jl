# Material Interface Implementation

# ============================================================================
# Emission / emissive checks (base fallbacks)
# With DiffuseAreaLight, no Material is ever emissive — emission lives on lights.
# ============================================================================

@propagate_inbounds is_emissive(::Material) = false
@propagate_inbounds is_pure_emissive(::Material) = false
@propagate_inbounds get_emission(::Material, ::Vec3f, ::Vec3f, ::Point2f) = RGBSpectrum(0f0)
@propagate_inbounds get_emission(::Material, ::Point2f) = RGBSpectrum(0f0)
