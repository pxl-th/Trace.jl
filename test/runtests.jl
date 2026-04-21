using Test
using Hikari
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using JET
using Lava

include("materials.jl")
include("type_stability.jl")
include("film.jl")
include("gpu_compat.jl")
include("volpath_integration.jl")
include("denoise.jl")
include("test_caching_gc_correctness.jl")
include("test_texture_wrap.jl")
include("test_hw_sw_parity.jl")
