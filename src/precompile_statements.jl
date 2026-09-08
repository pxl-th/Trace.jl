# GENERATED, do not hand-edit. Source: `--trace-compile=stderr
# --trace-compile-timing` over `out/trace_precompile.jl` — `parse_pbrt_string`
# of the scene at the bottom of Hikari.jl, then `build_hikari_scene` with the
# Lava backend, `hw_accel = true`, one sample — followed by filtering to the
# signatures that name Hikari/Raycore/the Vulkan backend and still resolve.
#
# Why this file exists: the `@compile_workload` at the bottom of Hikari.jl
# builds its scene on `KernelAbstractions.CPU()`, so nothing parameterised on
# the Vulkan backend is covered by it and gets re-inferred every session.
# `precompile` infers those without running them, so this stays device-free.
#
# `_precompile_statements` returns (succeeded, attempted). A stale signature
# returns `false` rather than failing, so `test/test_precompile_statements.jl`
# can assert the two are equal. Regenerate rather than delete when it fails: a
# signature that no longer matches usually means the real call site moved too.
#
# The backend's types live in MantleVulkanExt since 2026-08-28, and the
# extension only loads where Vulkan and Lava both do — not on a Mac, and not in
# a CI runner without a loader. The guard answers (0, 0) there, which the test
# accepts as "nothing to attempt"; on a Vulkan machine the extension is loaded
# by the time Hikari's own precompile reaches this file.
function _precompile_statements()
    ok = 0
    total = 0
    MVE = Base.get_extension(Mantle, :MantleVulkanExt)
    MVE === nothing && return (ok, total)
    total += 1; ok += precompile(Tuple{typeof(Core.kwcall), NamedTuple{(:backend, :samples, :max_depth, :hw_accel), Tuple{MVE.LavaBackend, Int64, Nothing, Bool}}, typeof(Hikari.build_hikari_scene), Hikari.PBRTScene})  # 3738.5 ms
    total += 1; ok += precompile(Tuple{Type{MVE.VkContext}})  # 1287.1 ms
    total += 1; ok += precompile(Tuple{typeof(Base.push!), Hikari.Scene{MVE.VulkanTLAS{Raycore.Triangle{Hikari.TriangleMeta}}, Raycore.MultiTypeSet{MVE.LavaBackend}, Raycore.MultiTypeSet{MVE.LavaBackend}, Raycore.MultiTypeSet{MVE.LavaBackend}, MVE.LavaArray{Hikari.MediumInterfaceIdx, 1}, Base.RefValue{Tuple{Raycore.Bounds3, GeometryBasics.HyperSphere{3, Float32}}}}, GeometryBasics.Mesh{3, Float32, GeometryBasics.NgonFace{3, Int64}, (:position, :normal, :uv), Tuple{Array{GeometryBasics.Point{3, Float32}, 1}, Array{Raycore.Normal{3, Float32}, 1}, Array{GeometryBasics.Point{2, Float32}, 1}}, Array{GeometryBasics.NgonFace{3, Int64}, 1}}, Hikari.Diffuse{Hikari.TexHandle, Hikari.TexHandle, Hikari.TexHandle}})  # 770.9 ms
    total += 1; ok += precompile(Tuple{typeof(Core.kwcall), NamedTuple{(:sbt_offset,), Tuple{UInt32}}, typeof(Base.push!), MVE.VulkanTLAS{Raycore.Triangle{Hikari.TriangleMeta}}, GeometryBasics.Mesh{3, Float32, GeometryBasics.NgonFace{3, GeometryBasics.OffsetInteger{-1, UInt32}}, (:position, :normal, :uv, :face_meta), Tuple{Array{GeometryBasics.Point{3, Float32}, 1}, Array{Raycore.Normal{3, Float32}, 1}, Array{GeometryBasics.Point{2, Float32}, 1}, GeometryBasics.FaceView{Hikari.TriangleMeta, Array{Hikari.TriangleMeta, 1}, Array{GeometryBasics.NgonFace{3, GeometryBasics.OffsetInteger{-1, UInt32}}, 1}}}, Array{GeometryBasics.NgonFace{3, GeometryBasics.OffsetInteger{-1, UInt32}}, 1}}, StaticArraysCore.SArray{Tuple{4, 4}, Float32, 2, 16}})  # 193.1 ms
    total += 1; ok += precompile(Tuple{Type{MVE.VkContext}, MVE.VK.Instance, MVE.VK.PhysicalDevice, MVE.VK.Device, UInt32, String, MVE.VK.Queue, MVE.VK.Queue, MVE.RTPipelineProperties, MVE.VK.DebugUtilsMessengerEXT, MVE.ValidationRing, Int64, Int64, UInt32, Int64, Bool, MVE.VK.PhysicalDeviceMemoryProperties, Tuple{Int64, Int64, Int64}, UInt64, Bool, Bool, Bool, Bool, Array{NamedTuple{(:M, :N, :K, :ab_type, :c_type, :scope), Tuple{Int64, Int64, Int64, UInt32, UInt32, UInt32}}, 1}, MVE.CoopMat2Caps, Bool, Bool, Bool, Bool, Bool, Bool, Bool, MVE.DebugConfig, String, Bool, MVE.VK.Queue, UInt32})  # 165.4 ms
    total += 1; ok += precompile(Tuple{Type{Hikari.PerspectiveCamera}, Raycore.Transformation, Raycore.Bounds2, Float32, Float32, Float32, Float32, Float32, Hikari.Film{MVE.LavaArray{ColorTypes.RGB{Float32}, 2}, MVE.LavaArray{ColorTypes.RGBA{Float32}, 2}, MVE.LavaArray{ColorTypes.RGB{Float32}, 2}, MVE.LavaArray{GeometryBasics.Vec{3, Float32}, 2}, MVE.LavaArray{Float32, 2}}})  # 162.3 ms
    total += 1; ok += precompile(Tuple{typeof(Base.push!), Hikari.Scene{MVE.VulkanTLAS{Raycore.Triangle{Hikari.TriangleMeta}}, Raycore.MultiTypeSet{MVE.LavaBackend}, Raycore.MultiTypeSet{MVE.LavaBackend}, Raycore.MultiTypeSet{MVE.LavaBackend}, MVE.LavaArray{Hikari.MediumInterfaceIdx, 1}, Base.RefValue{Tuple{Raycore.Bounds3, GeometryBasics.HyperSphere{3, Float32}}}}, Hikari.PointLight{Hikari.RGBIlluminantSpectrum}})  # 54.4 ms
    total += 1; ok += precompile(Tuple{Type{Mantle.Buffer{T, N} where N where T}, MVE.LavaDevice, Array{ColorTypes.RGB{Float32}, 2}})  # 13.9 ms
    total += 1; ok += precompile(Tuple{Type{Hikari.Film{FB, PP, AB, NB, DB} where DB<:AbstractArray{Float32, 2} where NB<:AbstractArray{GeometryBasics.Vec{3, Float32}, 2} where AB<:AbstractArray{ColorTypes.RGB{Float32}, 2} where PP<:AbstractArray{ColorTypes.RGBA{Float32}, 2} where FB<:AbstractArray{ColorTypes.RGB{Float32}, 2}}, MVE.LavaBackend, Hikari.Film{Array{ColorTypes.RGB{Float32}, 2}, Array{ColorTypes.RGBA{Float32}, 2}, Array{ColorTypes.RGB{Float32}, 2}, Array{GeometryBasics.Vec{3, Float32}, 2}, Array{Float32, 2}}})  # 13.7 ms
    total += 1; ok += precompile(Tuple{Type{Mantle.Buffer{T, N} where N where T}, MVE.LavaDevice, Array{GeometryBasics.Vec{3, Float32}, 2}})  # 12.0 ms
    total += 1; ok += precompile(Tuple{Type{Mantle.Buffer{T, N} where N where T}, MVE.LavaDevice, Array{ColorTypes.RGBA{Float32}, 2}})  # 11.9 ms
    total += 1; ok += precompile(Tuple{Type{Mantle.Buffer{T, N} where N where T}, MVE.LavaDevice, Array{Float32, 2}})  # 11.7 ms
    total += 1; ok += precompile(Tuple{typeof(Raycore.build_triangle), Array{GeometryBasics.Point{3, Float32}, 1}, Array{Raycore.Normal{3, Float32}, 1}, Array{GeometryBasics.Point{2, Float32}, 1}, Array{UInt32, 1}, Int64, Hikari.TriangleMeta})  # 8.2 ms
    total += 1; ok += precompile(Tuple{Type{StaticArraysCore.SArray{Tuple{3}, Raycore.Normal{3, Float32}, 1, 3}}, Raycore.Normal{3, Float32}, Vararg{Raycore.Normal{3, Float32}}})  # 4.8 ms
    total += 1; ok += precompile(Tuple{Type{Hikari.PBRTResult}, Hikari.Scene{MVE.VulkanTLAS{Raycore.Triangle{Hikari.TriangleMeta}}, Raycore.MultiTypeSet{MVE.LavaBackend}, Raycore.MultiTypeSet{MVE.LavaBackend}, Raycore.MultiTypeSet{MVE.LavaBackend}, MVE.LavaArray{Hikari.MediumInterfaceIdx, 1}, Base.RefValue{Tuple{Raycore.Bounds3, GeometryBasics.HyperSphere{3, Float32}}}}, Hikari.PerspectiveCamera, Hikari.Film{MVE.LavaArray{ColorTypes.RGB{Float32}, 2}, MVE.LavaArray{ColorTypes.RGBA{Float32}, 2}, MVE.LavaArray{ColorTypes.RGB{Float32}, 2}, MVE.LavaArray{GeometryBasics.Vec{3, Float32}, 2}, MVE.LavaArray{Float32, 2}}, NamedTuple{(:samples, :max_depth, :regularize, :russian_roulette_depth, :max_component_value), Tuple{Int64, Int64, Bool, Int64, Float32}}, Hikari.PixelSensor, String})  # 4.3 ms
    total += 1; ok += precompile(Tuple{Type{Raycore.Normal{3, Float32}}, Int64, Vararg{Int64}})  # 4.0 ms
    total += 1; ok += precompile(Tuple{typeof(Raycore.is_degenerate_face), Array{GeometryBasics.Point{3, Float32}, 1}, Array{UInt32, 1}, Int64})  # 3.9 ms
    total += 1; ok += precompile(Tuple{typeof(Hikari.parse_pbrt_string), String})  # 3.7 ms
    total += 1; ok += precompile(Tuple{Type{MVE.CoopMat2Caps}, Vararg{Bool, 8}})  # 3.1 ms
    total += 1; ok += precompile(Tuple{Type{Hikari.Film{FB, PP, AB, NB, DB} where DB<:AbstractArray{Float32, 2} where NB<:AbstractArray{GeometryBasics.Vec{3, Float32}, 2} where AB<:AbstractArray{ColorTypes.RGB{Float32}, 2} where PP<:AbstractArray{ColorTypes.RGBA{Float32}, 2} where FB<:AbstractArray{ColorTypes.RGB{Float32}, 2}}, GeometryBasics.Point{2, Float32}, Raycore.Bounds2, Float32, Array{Float32, 2}, Int32, GeometryBasics.Point{2, Float32}, Hikari.GPUFilterParams, Float32, MVE.LavaArray{ColorTypes.RGB{Float32}, 2}, MVE.LavaArray{ColorTypes.RGB{Float32}, 2}, MVE.LavaArray{GeometryBasics.Vec{3, Float32}, 2}, MVE.LavaArray{Float32, 2}, MVE.LavaArray{ColorTypes.RGBA{Float32}, 2}, Base.RefValue{Int32}, Base.RefValue{Any}, Hikari.DeviceMemory})  # 2.9 ms
    total += 1; ok += precompile(Tuple{Type{Raycore.Triangle{Hikari.TriangleMeta}}, StaticArraysCore.SArray{Tuple{3}, GeometryBasics.Point{3, Float32}, 1, 3}, StaticArraysCore.SArray{Tuple{3}, Raycore.Normal{3, Float32}, 1, 3}, StaticArraysCore.SArray{Tuple{3}, GeometryBasics.Vec{3, Float32}, 1, 3}, StaticArraysCore.SArray{Tuple{3}, GeometryBasics.Point{2, Float32}, 1, 3}, Hikari.TriangleMeta})  # 2.5 ms
    total += 1; ok += precompile(Tuple{Type{NamedTuple{(:backend, :samples, :max_depth, :hw_accel), T} where T<:Tuple}, Tuple{MVE.LavaBackend, Int64, Nothing, Bool}})  # 2.4 ms
    total += 1; ok += precompile(Tuple{typeof(Base.getindex), Array{Hikari.TriangleMeta, 1}, UInt32})  # 2.2 ms
    total += 1; ok += precompile(Tuple{Type{Mantle.AdaptedAccel{MVE.VulkanTLAS{Raycore.Triangle{Hikari.TriangleMeta}}, MVE.LavaArray{Raycore.Triangle{Hikari.TriangleMeta}, 1}, MVE.LavaArray{UInt32, 1}, Raycore.Triangle{Hikari.TriangleMeta}, Nothing}}, MVE.VulkanTLAS{Raycore.Triangle{Hikari.TriangleMeta}}, MVE.LavaArray{Raycore.Triangle{Hikari.TriangleMeta}, 1}, MVE.LavaArray{UInt32, 1}, Raycore.Triangle{Hikari.TriangleMeta}, Nothing})  # 2.2 ms
    total += 1; ok += precompile(Tuple{typeof(MVE.unsafe_free!), MVE.LavaArray{UInt32, 1}})  # 2.2 ms
    total += 1; ok += precompile(Tuple{typeof(MVE.unsafe_free!), MVE.LavaArray{Raycore.Triangle{Hikari.TriangleMeta}, 1}})  # 2.1 ms
    total += 1; ok += precompile(Tuple{typeof(MVE.unsafe_free!), MVE.LavaArray{UInt8, 1}})  # 2.1 ms
    total += 1; ok += precompile(Tuple{typeof(MVE.unsafe_free!), MVE.LavaArray{MVE.VulkanInstanceRecord, 1}})  # 2.1 ms
    total += 1; ok += precompile(Tuple{typeof(MVE.unsafe_free!), MVE.LavaArray{Hikari.Diffuse{Hikari.TexHandle, Hikari.TexHandle, Hikari.TexHandle}, 1}})  # 2.1 ms
    total += 1; ok += precompile(Tuple{typeof(MVE.unsafe_free!), MVE.LavaArray{Hikari.PointLight{Hikari.RGBIlluminantSpectrum}, 1}})  # 2.1 ms
    total += 1; ok += precompile(Tuple{typeof(MVE.unsafe_free!), MVE.LavaArray{Hikari.MediumInterfaceIdx, 1}})  # 2.1 ms
    total += 1; ok += precompile(Tuple{Type{Raycore.StaticMultiTypeSet{Data, Textures} where Textures<:Tuple where Data<:Tuple}, Tuple{MVE.LavaArray{Hikari.Diffuse{Hikari.TexHandle, Hikari.TexHandle, Hikari.TexHandle}, 1}}, Tuple{}})  # 1.8 ms
    total += 1; ok += precompile(Tuple{Type{StaticArrays.Args{T} where T<:Tuple}, Tuple{Raycore.Normal{3, Float32}, Raycore.Normal{3, Float32}, Raycore.Normal{3, Float32}}})  # 1.7 ms
    total += 1; ok += precompile(Tuple{Type{StaticArraysCore.SArray{Tuple{3}, Raycore.Normal{3, Float32}, 1, 3}}, Tuple{Raycore.Normal{3, Float32}, Raycore.Normal{3, Float32}, Raycore.Normal{3, Float32}}})  # 1.7 ms
    total += 1; ok += precompile(Tuple{Type{Hikari.TriangleMeta}, UInt32, UInt32, UInt32})  # 1.7 ms
    total += 1; ok += precompile(Tuple{Type{NamedTuple{(:raygen, :closest_hit, :miss, :payload_type), T} where T<:Tuple}, Tuple{typeof(MVE.hw_raygen), typeof(MVE.hw_closesthit), typeof(MVE.hw_miss), Symbol}})  # 1.7 ms
    total += 1; ok += precompile(Tuple{Type{Raycore.StaticMultiTypeSet{Data, Textures} where Textures<:Tuple where Data<:Tuple}, Tuple{MVE.LavaArray{Hikari.PointLight{Hikari.RGBIlluminantSpectrum}, 1}}, Tuple{}})  # 1.7 ms
    total += 1; ok += precompile(Tuple{typeof(Mantle.compatible), MVE.LavaDevice, UInt32, UInt32})  # 1.6 ms
    return (ok, total)
end
