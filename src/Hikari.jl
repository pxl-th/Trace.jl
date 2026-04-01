module Hikari

using Base: @propagate_inbounds
import FileIO
using ImageCore
using ImageIO
using GeometryBasics
using LinearAlgebra
using StaticArrays
using ProgressMeter
using StructArrays
using Atomix
using KernelAbstractions
using Raycore
using Zlib_jll
using Adapt
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA
using GPUArraysCore: @allowscalar
# Lava is a weak dependency — hw-rt.jl is loaded via ext/HikariLavaExt.jl

# Re-export Raycore types and functions that Trace uses
import Raycore: AbstractRay, Ray, RayDifferentials, apply, check_direction, scale_differentials
import Raycore: Bounds2, Bounds3, area, surface_area, diagonal, maximum_extent, offset, is_valid, inclusive_sides, expand
import Raycore: distance, distance_squared, bounding_sphere
# Note: lerp is defined in spectrum.jl for Spectrum, Float32, and Point3f
import Raycore: Transformation, translate, scale, rotate, rotate_x, rotate_y, rotate_z, look_at, perspective
import Raycore: swaps_handedness, has_scale
import Raycore: Triangle
import Raycore: TLAS, StaticTLAS, TraversableTLAS, TLASHandle, world_bound, closest_hit, any_hit, sync!

# Legacy alias - InstanceHandle was renamed to TLASHandle
const InstanceHandle = TLASHandle
import Raycore: Normal3f, intersect, intersect_p
import Raycore: is_dir_negative, increase_hit, intersect_p!
import Raycore: to_gpu
import Raycore: sum_unrolled, reduce_unrolled, for_unrolled, map_unrolled, getindex_unrolled
import Raycore: SetKey, MultiTypeSet, StaticMultiTypeSet, with_index, is_invalid, n_slots

abstract type Spectrum end
abstract type Light end
abstract type Material end
abstract type Integrator end
abstract type Medium end

# Default no-op close/clear for integrators without cached state
Base.close(::Integrator) = nothing
clear!(::Integrator) = nothing

const DO_ASSERTS = false
macro real_assert(expr, msg="")
    if DO_ASSERTS
        esc(:(@assert $expr $msg))
    else
        return :()
    end
end

include("spectrum.jl")
# PiecewiseLinearSpectrum needs SampledSpectrum/Wavelengths from spectral.jl,
# and must be available before texture-ref.jl and uber-material.jl
include("spectral/spectral.jl")
include("spectral/piecewise-linear.jl")
include("spectral/metal-spectra.jl")

include("random.jl")
include("surface_interaction.jl")
include("materials/medium-interface.jl")
include("scene.jl")

include("filter.jl")
include("film.jl")

include("camera/camera.jl")
include("sampler/sampling.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("textures/texture-ref.jl")
include("textures/environment_map.jl")

# Spectral rendering support
include("spectral/color.jl")
include("spectral/uplift.jl")
include("spectral/sensor.jl")
include("spectral/sensor_data.jl")

# Materials: shared math first, then each material, then dispatch
include("materials/common.jl")
include("materials/material.jl")
include("materials/diffuse.jl")
include("materials/dielectric.jl")
include("materials/conductor.jl")
include("materials/coated-diffuse.jl")
include("materials/mix-material.jl")
include("materials/coated-conductor.jl")
include("materials/coated-diffuse-transmission.jl")
include("materials/diffuse-transmission.jl")
include("materials/emissive.jl")
include("materials/dispatch.jl")

# Sobol sampler (needs mix_bits from materials/common.jl)
include("sampler/sobol_matrices.jl")
include("sampler/sobol.jl")
# Stratified sampler (needs murmur_hash_64a from spectral-eval.jl, sobol functions from sobol.jl)
include("sampler/stratified.jl")
include("primitive.jl")
include("lights/emission.jl")
include("lights/light.jl")
include("lights/point.jl")
include("lights/spot.jl")
include("lights/directional.jl")
include("lights/sun.jl")
include("lights/hosek_wilkie_data.jl")
include("lights/sun_sky.jl")
include("lights/ambient.jl")
include("lights/environment.jl")
include("lights/diffuse-area.jl")
include("lights/light-bounds.jl")
include("lights/light-sampler.jl")
include("lights/bvh-light-sampler.jl")
# GB.Mesh push! API (needs materials, lights, and DiffuseAreaLight)
include("scene-mesh.jl")
# Unified work queue for wavefront integrators
include("integrators/workqueue.jl")
# Spectral light sampling (used by VolPath for direct lighting, environment evaluation)
include("lights/spectral-sampling.jl")
# VolPath volumetric path tracer
include("integrators/volpath/media.jl")
include("integrators/volpath/nanovdb.jl")
include("integrators/volpath/medium-dispatch.jl")
include("integrators/volpath/workitems.jl")
include("integrators/volpath/volpath-state.jl")
include("integrators/volpath/delta-tracking.jl")
include("integrators/volpath/medium-scatter.jl")
include("integrators/volpath/intersection.jl")
include("integrators/volpath/surface-eval.jl")
include("integrators/volpath/volpath.jl")
# Hardware RT integration (HWTLAS, HWAdaptedAccel, dispatch overrides)
# Loaded via ext/HikariLavaExt.jl when Lava is available
include("kernel-abstractions.jl")
# Postprocessing pipeline
include("postprocess.jl")

# Denoising
include("denoise.jl")

# PBRT file parser
include("pbrt/tokenizer.jl")
include("pbrt/parser.jl")
include("pbrt/scene_builder.jl")

# include("model_loader.jl")

end
