# Edge-Avoiding À-Trous Wavelet Denoiser for Path Tracing
# Based on SVGF (Schied et al. 2017) and EAW (Dammertz et al. 2010)
#
# Uses auxiliary buffers (albedo, normals, depth) from Film for edge-stopping
# to preserve sharp features while removing Monte Carlo noise.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA
using StaticArrays

# =============================================================================
# Denoiser Configuration
# =============================================================================

"""
    DenoiseConfig

Configuration parameters for the à-trous wavelet denoiser.

# Fields
- `iterations`: Number of filter passes (each doubles the filter radius)
- `sigma_color`: Color edge-stopping threshold (luminance sensitivity)
- `sigma_normal`: Normal edge-stopping threshold (angular sensitivity)
- `sigma_depth`: Depth edge-stopping threshold (distance sensitivity)
- `use_variance`: Whether to use per-pixel variance to guide filtering
"""
struct DenoiseConfig
    iterations::Int32       # Number of à-trous iterations (typically 4-5)
    sigma_color::Float32    # Color/luminance edge-stopping
    sigma_normal::Float32   # Normal edge-stopping (higher = more blur across normals)
    sigma_depth::Float32    # Depth edge-stopping (higher = more blur across depth)
    use_variance::Bool      # Use variance-guided filtering (SVGF style)
end

"""
    DenoiseConfig(; iterations=4, sigma_color=1.0, sigma_normal=64.0,
                    sigma_depth=0.1, use_variance=false)

Create a denoiser configuration with sensible defaults.

Units (post log-luminance / relative-depth fix):
- `sigma_color` is in **stops** (log2 luminance diff) — 1.0 means a 2× ratio
  weighted at `exp(-1) ≈ 0.37`. Lower = sharper edges.
- `sigma_normal` is the exponent on `clamp(dot(n_p, n_q), 0, 1)`. Higher =
  more sensitive to normal differences.
- `sigma_depth` is **relative** depth tolerance (fraction of local depth) —
  0.1 means a 10% depth diff per filter step weighted at `exp(-1)`.
- `use_variance` is currently a no-op (see `weight_color` doc).
"""
function DenoiseConfig(;
    iterations::Int=4,
    sigma_color::Real=1.0f0,
    sigma_normal::Real=64.0f0,
    sigma_depth::Real=0.1f0,
    use_variance::Bool=false
)
    return DenoiseConfig(
        Int32(iterations),
        Float32(sigma_color),
        Float32(sigma_normal),
        Float32(sigma_depth),
        use_variance
    )
end

# =============================================================================
# Edge-Stopping Weight Functions
# =============================================================================

"""
    denoise_luminance(r, g, b) -> Float32

Compute luminance from RGB using Rec. 709 coefficients.
"""
@propagate_inbounds function denoise_luminance(r::Float32, g::Float32, b::Float32)::Float32
    return 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
end

"""
    weight_color(lum_p, lum_q, sigma, variance) -> Float32

Color/luminance edge-stopping weight in **log-luminance space** (HDR-safe).
`sigma` is in *stops* (log2 luminance ratio) — a 2^sigma ratio between p and q
weights to `exp(-1) ≈ 0.37`.

**Asymmetric:** when `q` is dimmer than the center `p` we treat the diff as
`FIREFLY_RATIO ≈ 0.3×` of its actual magnitude, so bright outliers pool down
into their dim surroundings (firefly suppression). When `q` is brighter we
stay strict so genuine edges viewed from the dim side are preserved. This is
the standard fix for à-trous's intrinsic inability to remove fireflies via a
purely symmetric bilateral weight — a firefly otherwise looks like an edge.

`variance` is ignored (kept for API): real SVGF needs *temporal* variance
from sample history, which this `Film` does not store. Using spatial variance
instead was actively harmful (it mistakes texture for noise).
"""
const FIREFLY_RATIO = 0.1f0

@propagate_inbounds function weight_color(
    lum_p::Float32, lum_q::Float32,
    sigma::Float32, _variance::Float32
)::Float32
    log_p = log2(max(lum_p, 0.0f0) + 1.0f-4)
    log_q = log2(max(lum_q, 0.0f0) + 1.0f-4)
    signed_diff = log_p - log_q   # > 0 ⇒ q is dimmer than p
    diff = signed_diff > 0.0f0 ? signed_diff * FIREFLY_RATIO : -signed_diff
    return exp(-diff / max(sigma, 1.0f-4))
end

"""
    weight_normal(n_p, n_q, sigma) -> Float32

Normal edge-stopping weight using dot product.
High sigma means more tolerance for normal differences.
"""
@propagate_inbounds function weight_normal(
    n_p::Vec3f, n_q::Vec3f, sigma::Float32
)::Float32
    dot_val = dot(n_p, n_q)
    # Clamp to [0, 1] and apply power
    dot_clamped = max(0.0f0, dot_val)
    return dot_clamped ^ sigma
end

"""
    weight_depth(d_p, d_q, sigma, step_size) -> Float32

Depth edge-stopping weight using a **relative** depth difference so the same
`sigma` works at any scene scale. `sigma` is the fractional depth tolerance
(e.g. `0.1` = 10% of local depth). Scales with `step_size` because the
expected per-step depth delta on an angled surface grows with filter radius.

Standard SVGF/EAW would use the actual screen-space depth gradient `∇d`
here; we approximate it by `max(d_p, d_q) * step_size`, which is correct up
to a constant for perspective surfaces and never blows up for misses.
"""
@propagate_inbounds function weight_depth(
    d_p::Float32, d_q::Float32,
    sigma::Float32, step_size::Float32
)::Float32
    diff = abs(d_p - d_q)
    scale = max(d_p, d_q, 1.0f0)
    return exp(-diff / (sigma * step_size * scale + 1.0f-4))
end

# =============================================================================
# À-Trous Wavelet Kernel
# =============================================================================

# 5x5 B-spline wavelet kernel weights: h = [1/16, 1/4, 3/8, 1/4, 1/16].
@inline function atrous_kernel_1d(i::Int32)
    kern = SVector{5,Float32}(0.0625f0, 0.25f0, 0.375f0, 0.25f0, 0.0625f0)
    return kern[i]
end

"""
    atrous_denoise_kernel!(output, input, normals, depth, variance,
                           width, height, step_size, config)

Single pass of the à-trous wavelet filter.
Applies a 5x5 filter with edge-stopping weights.
"""
@kernel inbounds=true function atrous_denoise_kernel!(
    output,  # RGB{Float32} matrix (height × width)
    @Const(input),   # RGB{Float32} matrix
    @Const(normals), # Vec3f matrix
    @Const(depth),   # Float32 matrix
    @Const(variance), # Float32 matrix (can be nothing-like placeholder)
    @Const(width::Int32), @Const(height::Int32),
    @Const(step_size::Int32),
    @Const(sigma_color::Float32),
    @Const(sigma_normal::Float32),
    @Const(sigma_depth::Float32),
    @Const(use_variance::Bool)
)
    idx = @index(Global)
    num_pixels = width * height

     if idx <= num_pixels
        # Convert linear index to 2D coordinates (row, col) for Julia matrices
        row = ((idx - Int32(1)) % height) + Int32(1)
        col = ((idx - Int32(1)) ÷ height) + Int32(1)

        # Get center pixel data
        pixel_p = input[row, col]
        r_p, g_p, b_p = pixel_p.r, pixel_p.g, pixel_p.b
        lum_p = denoise_luminance(r_p, g_p, b_p)

        n_p = normals[row, col]
        d_p = depth[row, col]
        var_p = use_variance ? variance[row, col] : 0.0f0

        # Accumulate filtered result
        sum_r = 0.0f0
        sum_g = 0.0f0
        sum_b = 0.0f0
        sum_weight = 0.0f0

        # 5x5 filter with step_size spacing
        for dy_i in Int32(1):Int32(5)
            for dx_i in Int32(1):Int32(5)
                dy = dy_i - Int32(3)  # -2 to 2
                dx = dx_i - Int32(3)

                # Neighbor coordinates
                q_row = row + dy * step_size
                q_col = col + dx * step_size

                # Boundary check (clamp to edge)
                q_row = clamp(q_row, Int32(1), height)
                q_col = clamp(q_col, Int32(1), width)

                # Get neighbor data
                pixel_q = input[q_row, q_col]
                r_q, g_q, b_q = pixel_q.r, pixel_q.g, pixel_q.b
                lum_q = denoise_luminance(r_q, g_q, b_q)

                n_q = normals[q_row, q_col]
                d_q = depth[q_row, q_col]

                # Compute spatial kernel weight (2D separable B-spline)
                k_x = atrous_kernel_1d(dx_i)
                k_y = atrous_kernel_1d(dy_i)
                w_spatial = k_x * k_y

                # Edge-stopping weights
                w_color = weight_color(lum_p, lum_q, sigma_color, var_p)
                w_norm = weight_normal(n_p, n_q, sigma_normal)
                w_depth = weight_depth(d_p, d_q, sigma_depth, Float32(step_size))

                # Combined weight
                weight = w_spatial * w_color * w_norm * w_depth

                # Accumulate
                sum_r += r_q * weight
                sum_g += g_q * weight
                sum_b += b_q * weight
                sum_weight += weight
            end
        end

        # Normalize and store result
        if sum_weight > 1.0f-6
            inv_w = 1.0f0 / sum_weight
            output[row, col] = RGB{Float32}(sum_r * inv_w, sum_g * inv_w, sum_b * inv_w)
        else
            # No valid neighbors - keep original
            output[row, col] = RGB{Float32}(r_p, g_p, b_p)
        end
    end
end

# =============================================================================
# Variance Computation
# =============================================================================

"""
    compute_variance_kernel!(variance, input, width, height)

Compute per-pixel variance from RGB framebuffer.
Uses spatial 3x3 neighborhood for variance estimation.
"""
@kernel inbounds=true function compute_variance_kernel!(
    variance,  # Float32 matrix
    @Const(input),  # RGB{Float32} matrix
    @Const(width::Int32), @Const(height::Int32)
)
    idx = @index(Global)
    num_pixels = width * height

     if idx <= num_pixels
        # Convert linear index to 2D
        row = ((idx - Int32(1)) % height) + Int32(1)
        col = ((idx - Int32(1)) ÷ height) + Int32(1)

        # Compute spatial variance over 3x3 neighborhood
        sum_lum = 0.0f0
        sum_lum_sq = 0.0f0
        count = Int32(0)

        for dy in Int32(-1):Int32(1)
            for dx in Int32(-1):Int32(1)
                q_row = row + dy
                q_col = col + dx

                if q_row >= Int32(1) && q_row <= height && q_col >= Int32(1) && q_col <= width
                    pixel = input[q_row, q_col]
                    lum = denoise_luminance(pixel.r, pixel.g, pixel.b)
                    sum_lum += lum
                    sum_lum_sq += lum * lum
                    count += Int32(1)
                end
            end
        end

        # Variance = E[X²] - E[X]²
        if count > Int32(0)
            mean = sum_lum / Float32(count)
            mean_sq = sum_lum_sq / Float32(count)
            var = max(0.0f0, mean_sq - mean * mean)
            variance[row, col] = var
        else
            variance[row, col] = 0.0f0
        end
    end
end

# =============================================================================
# High-Level Denoising API (works with Film)
# =============================================================================

"""
    denoise!(film::Film; config=DenoiseConfig())

Apply edge-avoiding à-trous wavelet denoising to `film.framebuffer` in place.
Uses `film.normal` and `film.depth` as edge-stopping guides.

# Arguments
- `film`: Film with framebuffer + aux buffers populated
- `config`: DenoiseConfig with filter parameters

# Notes
- Requires `film.normal` and `film.depth` populated (e.g. via `fill_aux_buffers!`).
- Mutates `film.framebuffer` (downstream `postprocess!` reads it).
- `config.use_variance` is currently a no-op: see `weight_color` docstring.
"""
function denoise!(film::Film; config::DenoiseConfig=DenoiseConfig())
    height, width = size(film.framebuffer)
    num_pixels = width * height
    backend = KA.get_backend(film.framebuffer)

    # use_variance is intentionally ignored; pass a zero-length placeholder so
    # the kernel signature stays stable. The kernel never reads it when
    # use_variance==false, but KA still wants a typed array argument.
    variance_placeholder = similar(film.depth, 0)

    # Ping-pong: buffer_a aliases the framebuffer (input on iter 1, also final
    # destination on even-iter counts). buffer_b is a scratch.
    buffer_a = film.framebuffer
    buffer_b = similar(film.framebuffer)

    denoise_kernel! = atrous_denoise_kernel!(backend)

    for i in 1:config.iterations
        step_size = Int32(1 << (i - 1))  # 1, 2, 4, 8, 16…
        if i % 2 == 1
            denoise_kernel!(
                buffer_b, buffer_a, film.normal, film.depth, variance_placeholder,
                Int32(width), Int32(height), step_size,
                config.sigma_color, config.sigma_normal, config.sigma_depth,
                false; ndrange=num_pixels)
        else
            denoise_kernel!(
                buffer_a, buffer_b, film.normal, film.depth, variance_placeholder,
                Int32(width), Int32(height), step_size,
                config.sigma_color, config.sigma_normal, config.sigma_depth,
                false; ndrange=num_pixels)
        end
        KA.synchronize(backend)
    end

    # Ensure the final result lives in film.framebuffer.
    if config.iterations % 2 == 1
        film.framebuffer .= buffer_b
    end
    return nothing
end
