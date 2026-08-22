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
# Lava is a hard dependency: `hw-rt.jl` below imports it unconditionally.
# (An earlier comment here called it weak and pointed at an ext that does not
# exist — the include at the bottom of this file has never been conditional.)
import Mantle

# Unexported but API: RayMakie's pbrt bridge calls it, so it is not free to
# change shape. `public` says that without putting it in every `using`'s scope.
public attach_bump

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

# Where a render state's device memory comes from. First, because everything
# below that allocates on the device takes a `DeviceMemory` in its signature —
# the spectral tables, the light BVH, the Sobol matrices, the work queues.
include("device-memory.jl")

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
include("textures/tex-handle.jl")
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
include("materials/bump-mapped.jl")
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
include("integrators/volpath/per_material_queues.jl")
include("integrators/volpath/volpath-state.jl")
include("integrators/volpath/delta-tracking.jl")
include("integrators/volpath/medium-scatter.jl")
include("integrators/volpath/intersection.jl")
include("integrators/volpath/surface-eval.jl")
# The sample as Mantle graphs — needs every stage's kernel to exist
include("integrators/volpath/graph.jl")
include("integrators/volpath/volpath.jl")
# Hardware RT dispatch (uses Raycore's backend-agnostic HWTLAS/HWAdaptedAccel interface)
include("integrators/volpath/hw-rt.jl")
# RT pipeline (raygen+closesthit+miss via VkRayTracingPipelineKHR + SBT) variant
include("integrators/volpath/rt-pipeline.jl")
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

# ── Precompile workload ─────────────────────────────────────────────────────
#
# Hikari had no workload at all, so the first scene built in a session paid to
# JIT the parser and the whole scene builder. Measured on a 128x128 pbrt scene:
# 12.8 s for the first `load_pbrt`, 0.04 s for the second, i.e. essentially all
# of it was first-call latency rather than work.
#
# DEVICE-FREE: `build_hikari_scene` defaults to a CPU backend, so this
# never touches the Vulkan driver — precompilation must not, and a device crash
# here has poisoned pkgimages before (see Lava's `__init__`).
#
# A CPU-backend build cannot specialise the Lava-array half of the builder, so
# this does not eliminate the cost, it removes the backend-independent part:
# parsing, material and texture construction, the spectral tables and the BVH.
# Measured: parse 0.8 s + CPU build 5.3 s up front cuts the subsequent Lava-backed
# build from 12.8 s to 9.3 s.
#
# `Lava.@setup_workload`/`@compile_workload` are PrecompileTools' macros
# re-exported by Lava (the latter also wrapping Lava's frozen-kernel recording,
# a no-op here since a CPU build compiles no SPIR-V), which is why Hikari needs
# no direct PrecompileTools dependency.
const _PRECOMPILE_SCENE = """
Film "rgb" "integer xresolution" 16 "integer yresolution" 16
LookAt 0 -1.2 0.6   0 0 0.5   0 0 1
Camera "perspective" "float fov" 40
Integrator "volpath" "integer maxdepth" 3
WorldBegin
LightSource "point" "rgb I" [40 40 40] "point3 from" [2 -1.5 3]
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "trianglemesh"
  "point3 P" [ -1 0 -1  1 0 -1  1 0 1  -1 0 1 ]
  "normal N" [ 0 -1 0  0 -1 0  0 -1 0  0 -1 0 ]
  "point2 uv" [ 0 0  1 0  1 1  0 1 ]
  "integer indices" [ 0 1 2  0 2 3 ]
"""

Lava.@setup_workload begin
    Lava.@compile_workload "hikari_scene_1" begin
        pbrt = parse_pbrt_string(_PRECOMPILE_SCENE)
        build_hikari_scene(pbrt; backend = KernelAbstractions.CPU(), samples = 1,
                           max_depth = nothing, hw_accel = false)
    end
end

end
