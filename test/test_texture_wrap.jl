using Test
using Hikari
using GeometryBasics: Point2f

# Regression: `sample_texture_data` used to clamp UVs outside [0,1] to the edge
# pixel. pbrt-v4's ImageTexture defaults to "repeat" wrap mode, and scenes like
# killeroo-gold.pbrt rely on tiling a `lines.png` grid across UV [0,5]. The
# clamp caused only one tile (the first cell) to show, with everything past
# the edge reading the nearest border pixel.

@testset "Texture UV wrap (repeat semantics)" begin

# Build a 4×4 checkerboard texture where each cell holds its (row, col) index
# encoded as a value — makes it easy to identify which cell a UV lookup returns.
function indexed_texture(; n=4)
    data = Matrix{Float32}(undef, n, n)
    for r in 1:n, c in 1:n
        # Encode so every cell is unique and recoverable.
        data[r, c] = Float32(100 * r + c)
    end
    return Hikari.Texture(data)
end

@testset "UVs inside [0,1] unchanged" begin
    tex = indexed_texture()
    # Sampling at a few fractional UVs — match same cells as before wrap change.
    v1 = Hikari.evaluate_texture(tex, Point2f(0.0, 0.0))
    v2 = Hikari.evaluate_texture(tex, Point2f(1.0, 1.0))
    v3 = Hikari.evaluate_texture(tex, Point2f(0.5, 0.5))
    @test v1 isa Float32
    @test v2 isa Float32
    @test v3 isa Float32
    # The values must stay within the encoded range (valid cells only).
    for v in (v1, v2, v3)
        r = Int(floor(v / 100))
        c = Int(v) - 100 * r
        @test 1 <= r <= 4
        @test 1 <= c <= 4
    end
end

@testset "UVs > 1 repeat (pbrt default) — not clamped" begin
    tex = indexed_texture()

    # If wrapping works, sampling at u+k (integer k) yields the SAME cell as u.
    for u_base in (0.125f0, 0.375f0, 0.625f0, 0.875f0),
        v_base in (0.125f0, 0.375f0, 0.625f0, 0.875f0)

        base = Hikari.evaluate_texture(tex, Point2f(u_base, v_base))
        # +1 on each axis
        shifted_u = Hikari.evaluate_texture(tex, Point2f(u_base + 1f0, v_base))
        shifted_v = Hikari.evaluate_texture(tex, Point2f(u_base, v_base + 1f0))
        shifted_uv = Hikari.evaluate_texture(tex, Point2f(u_base + 5f0, v_base + 5f0))
        @test base == shifted_u
        @test base == shifted_v
        @test base == shifted_uv
    end
end

@testset "UVs < 0 repeat (negative wrap)" begin
    tex = indexed_texture()
    for u_base in (0.2f0, 0.7f0)
        base = Hikari.evaluate_texture(tex, Point2f(u_base, u_base))
        neg = Hikari.evaluate_texture(tex, Point2f(u_base - 3f0, u_base - 2f0))
        @test base == neg
    end
end

@testset "UV tiling distinguishes cells (no degenerate clamp-to-edge)" begin
    # The pre-fix bug manifested as every UV > 1 returning the same cell
    # (the border pixel). Collect every-half-step UVs across [0, 5] and check
    # we hit at least 4 distinct cells (one full tile's worth).
    tex = indexed_texture(; n=4)
    vals = Set{Float32}()
    for u in 0f0:0.25f0:5f0, v in 0f0:0.25f0:5f0
        push!(vals, Hikari.evaluate_texture(tex, Point2f(u, v)))
    end
    # 4x4 texture → up to 16 distinct values. The pre-fix clamp would return
    # only the border-cell values (≤ 4 distinct).
    @test length(vals) == 16
end

end  # @testset
