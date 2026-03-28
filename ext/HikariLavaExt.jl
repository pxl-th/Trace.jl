module HikariLavaExt

using Hikari
using Lava

using Base: @propagate_inbounds

# Re-import Hikari internals that hw-rt.jl extends or uses
import Hikari: _default_accel, _gpu_ndrange, _detect_initial_medium,
               vp_trace_rays!, vp_trace_shadow_rays!, vp_trace_rays_kernel!,
               fill_aux_buffers!, sync!,
               Scene, Film, VolPath, VolPathState,
               SampledSpectrum, SampledWavelengths,
               has_medium, is_medium_transition, get_crossing_medium,
               compute_transmittance_ratio_tracking,
               vp_compute_geometric_normal, vp_compute_uv_barycentric,
               get_surface_alpha_dispatch, pbrt_hash, pcg32_init, pcg32_uniform_f32,
               is_black, average, accumulate_spectrum!,
               CameraSample, generate_ray

import Raycore
import Raycore: TLASHandle, Bounds3, Normal3f, bounding_sphere,
                build_triangle, is_degenerate_face
import Adapt
import KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
using Atomix
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, Point2f, decompose, decompose_normals, TriangleFace, GLTriangleFace
using StaticArrays: SVector, SMatrix
using LinearAlgebra: I, dot
using ImageCore: RGB

# Include the hw-rt implementation
include(joinpath(@__DIR__, "..", "src", "integrators", "volpath", "hw-rt.jl"))

end
