@testset "Fresnel Dielectric" begin
    # Vacuum gives no reflectance.
    @test Hikari.fresnel_dielectric(1f0, 1f0, 1f0) ≈ 0f0
    @test Hikari.fresnel_dielectric(0.5f0, 1f0, 1f0) ≈ 0f0
end

# NOTE: the remainder of this file used to test the pre-Material-wrapper
# BxDF objects (`SpecularReflection`, `SpecularTransmission`,
# `MicrofacetReflection`, `MicrofacetTransmission`, `FresnelSpecular`,
# `fresnel_conductor`) and the `BSDF_*` flag constants. Those were removed
# when Hikari moved to the Material wrappers (`Diffuse`, `Conductor`,
# `Dielectric`, `CoatedDiffuse`, `CoatedConductor`, …). Per-material
# BSDF sampling is exercised end-to-end by `test/pbrt/test_pbrt_all_
# materials.jl` against pbrt-v4 reference EXRs, which gives stronger
# coverage than re-implementing unit tests against the new wrappers'
# internal `sample_bsdf_spectral` / `evaluate_bsdf_spectral` methods.
