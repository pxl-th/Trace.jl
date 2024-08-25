
"""
Point in (x, y) format.
"""
@inline function get_pixel_index(crop_bounds, p::Point2)
    i1, i2 = u_int32.((p .- crop_bounds.p_min .+ 1.0f0))
    return CartesianIndex(i2, i1)
end


function merge_film_tile!(f::AbstractMatrix{Pixel}, crop_bounds::Bounds2, ft::AbstractMatrix{Pixel}, tile::Bounds2, tile_col::Int32)
    ft_contrib_sum = ft.contrib_sum
    ft_filter_weight_sum = ft.filter_weight_sum
    f_xyz = f.xyz
    f_filter_weight_sum = f.filter_weight_sum
    linear = Int32(1)
    @inbounds for pixel in tile
        f_idx = get_pixel_index(crop_bounds, pixel)
        f_xyz[f_idx] += to_XYZ(ft_contrib_sum[linear, tile_col])
        f_filter_weight_sum[f_idx] += ft_filter_weight_sum[linear, tile_col]
        linear += Int32(1)
    end
    return
end


@inline function get_tile_index(bounds::Bounds2, p::Point2, column)
    i, j = u_int32.((p .- bounds.p_min .+ 1.0f0))
    ncols = Int32(inclusive_sides(bounds)[1])
    row = (i - 1) * ncols + j
    return CartesianIndex(row, column)
end


function add_sample!(
    tiles::AbstractMatrix{Pixel}, tile::Bounds2, tile_column, point::Point2f, spectrum::S,
    filter_table, filter_radius, inv_filter_radius, filter_table_width,
    sample_weight::Float32=1.0f0,
) where {S<:Spectrum}

    # Compute sample's raster bounds.
    discrete_point = point .- 0.5f0
    # Compute sample radius around point
    p0 = u_int32.(ceil.(discrete_point .- filter_radius))
    p1 = u_int32.(floor.(discrete_point .+ filter_radius)) .+ Int32(1)
    # Make sure we're inbounds
    p0 = Int32.(max.(p0, max.(tile.p_min, Point2{Int32}(1))))
    p1 = Int32.(min.(p1, tile.p_max))
    # Precompute x & y filter offsets.
    offsets_x = filter_offsets(p0[1], p1[1], discrete_point, inv_filter_radius[1], filter_table_width)
    offsets_y = filter_offsets(p0[2], p1[2], discrete_point, inv_filter_radius[2], filter_table_width)
    # Loop over filter support & add sample to pixel array.
    contrib_sum = tiles.contrib_sum
    filter_weight_sum = tiles.filter_weight_sum
    xrange = p0[1]:p1[1]
    yrange = p0[2]:p1[2]
    @inbounds for (j, y) in enumerate(xrange), (i, x) in enumerate(yrange)
        w = filter_table[offsets_y[j], offsets_x[i]]
        @real_assert sample_weight <= 1
        @real_assert w <= 1
        idx = get_tile_index(tile, Point2(x, y), tile_column)
        contrib_sum[idx] += spectrum * sample_weight * w
        filter_weight_sum[idx] += w
    end
end
