using Test
using Hikari
using GeometryBasics: Point2f

# CheckerboardTexture is an exact port of pbrt-v4's CheckerboardTexture
# (textures.cpp `Checkerboard()`, 2D case, fused with UVMapping::Map). It
# replaced the 256² LUT rasterization in the pbrt scene builder: the LUT's
# bilinear smoothing both quantized checker-edge positions and widened the
# height-gradient bands that bump mapping differentiates, which biased
# shadow_bumpgold_dome_over_velvet 21% dark at 256 spp.
#
# These tests pin the analytic evaluation against hand-computed pbrt values so
# the LUT (or any other approximation) can't sneak back in.

# Reference implementation of pbrt-v4's helpers, straight from textures.cpp.
pbrt_d(x) = begin
    y = x / 2 - floor(x / 2) - 0.5
    x / 2 + y * (1 - 2 * abs(y))
end
pbrt_bf(x, r) = begin
    if floor(x - r) == floor(x + r)
        return 1.0 - 2.0 * (Int(floor(x)) & 1)
    end
    (pbrt_d(x + r) - 2 * pbrt_d(x) + pbrt_d(x - r)) / (r * r)
end
function pbrt_checker_weight(u, v, su, sv, du, dv, dudx, dudy, dvdx, dvdy)
    s = su * u + du
    t = sv * v + dv
    ds = 1.5 * max(abs(su * dudx), abs(su * dudy))
    dt = 1.5 * max(abs(sv * dvdx), abs(sv * dvdy))
    return 0.5 - 0.5 * pbrt_bf(s, ds) * pbrt_bf(t, dt)
end

@testset "CheckerboardTexture (pbrt-v4 analytic port)" begin

@testset "point-sample parity (zero derivatives)" begin
    # 8×8 checkerboard over [0,1]²: even cell parity → tex1.
    cb = Hikari.CheckerboardTexture(8f0, 8f0, 0f0, 0f0, 10f0, 20f0)
    for (u, v, expected) in [
        (0.01f0, 0.01f0, 10f0),   # cell (0,0) even
        (0.13f0, 0.01f0, 20f0),   # cell (1,0) odd
        (0.13f0, 0.13f0, 10f0),   # cell (1,1) even
        (0.99f0, 0.99f0, 10f0),   # cell (7,7) even
        (0.99f0, 0.87f0, 20f0),   # cell (7,6) odd
    ]
        w = Hikari.checker_bf(8f0 * u, 0f0) * Hikari.checker_bf(8f0 * v, 0f0)
        val = expected == 10f0 ? 1f0 : -1f0
        @test w == val
    end
end

@testset "checker_bf matches pbrt reference" begin
    # atol accounts for Float32 cancellation in the (d(x+r) - 2d(x) + d(x-r))/r²
    # second difference at large |x| — pbrt's own Float32 evaluation has the
    # same error; the Float64 reference here is just more precise.
    for x in Float32[-3.7, -0.2, 0.0, 0.4, 1.3, 2.5, 7.99], r in Float32[0.0, 0.01, 0.3, 1.7]
        @test Hikari.checker_bf(x, r) ≈ Float32(pbrt_bf(x, r)) atol = 2e-3
    end
end

@testset "filtered weight matches pbrt reference" begin
    for (u, v) in [(0.124f0, 0.5f0), (0.5f0, 0.126f0), (0.49f0, 0.51f0)]
        for deriv in Float32[0.0, 0.001, 0.02, 0.2]
            expected = Float32(pbrt_checker_weight(u, v, 8.0, 8.0, 0.0, 0.0,
                                                   deriv, 0.0, 0.0, deriv))
            cb = Hikari.CheckerboardTexture(8f0, 8f0, 0f0, 0f0, 0f0, 1f0)
            s = cb.su * u + cb.du
            t = cb.sv * v + cb.dv
            ds = 1.5f0 * max(abs(cb.su * deriv), abs(cb.su * 0f0))
            dt = 1.5f0 * max(abs(cb.sv * 0f0), abs(cb.sv * deriv))
            w = 0.5f0 - 0.5f0 * Hikari.checker_bf(s, ds) * Hikari.checker_bf(t, dt)
            @test w ≈ expected atol = 1e-5
            # Wide filters converge to the 0.5 average.
            deriv > 0.1f0 && @test abs(w - 0.5f0) < 0.2f0
        end
    end
end

@testset "udelta/vdelta shift the lattice" begin
    # Shifting by exactly one cell flips parity.
    w0 = Hikari.checker_bf(1f0 * 0.3f0 + 0f0, 0f0)
    w1 = Hikari.checker_bf(1f0 * 0.3f0 + 1f0, 0f0)
    @test w0 == -w1
end

end

@testset "pbrt imagemap encoding rule (PNG → sRGB default)" begin
    # pbrt-v4 textures.cpp:436: the DEFAULT encoding is extension-based —
    # `.png → sRGB, everything else → linear` — for float AND spectrum
    # classes, including 16-bit PNGs (image.cpp ReadPNG decodes 16-bit
    # through `encoding.ToFloatLinear`). Hikari used to sRGB-decode only
    # 8-bit spectrum-class PNGs, which left float height maps linear and
    # scaled bump gradients wrong by the local slope of the sRGB curve
    # (~1.45× at h≈0.7) — pinned by tex_conductor_bumpmap_light_point.
    @test Hikari.srgb_to_linear(0.04045f0) ≈ 0.04045f0 / 12.92f0
    @test Hikari.srgb_to_linear(0.5f0) ≈ ((0.5f0 + 0.055f0) / 1.055f0)^2.4f0
    @test Hikari.srgb_to_linear(1f0) ≈ 1f0
end
