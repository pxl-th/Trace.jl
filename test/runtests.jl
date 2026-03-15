using Test
using Hikari
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using JET

include("materials.jl")
include("type_stability.jl")
include("film.jl")
include("gpu_compat.jl")
include("volpath_integration.jl")

# GPU tests (require Lava/Vulkan device)
if haskey(ENV, "HIKARI_GPU_TESTS") || try using Lava; Lava.vk_context(); true catch; false end
    include("test_caching_gc_correctness.jl")
end
