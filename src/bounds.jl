struct Bounds2
    p_min::Point2f0
    p_max::Point2f0
end

struct Bounds3
    p_min::Point3f0
    p_max::Point3f0
end

# By default -- create bounds in invalid configuraiton.
Bounds2() = Bounds2(Point2f0(Inf32), Point2f0(-Inf32))
Bounds3() = Bounds3(Point3f0(Inf32), Point3f0(-Inf32))
# Encapsulate single point.
Bounds2(p::Point2f0) = Bounds2(p, p)
Bounds3(p::Point3f0) = Bounds3(p, p)
# Construct correctly from two given points.
Bounds2c(p1::Point2f0, p2::Point2f0) = Bounds2(min.(p1, p2), max.(p1, p2))
Bounds3c(p1::Point3f0, p2::Point3f0) = Bounds3(min.(p1, p2), max.(p1, p2))

function Base.:(==)(b1::Union{Bounds2, Bounds3}, b2::Union{Bounds2, Bounds3})
    b1.p_min == b2.p_min && b1.p_max == b2.p_max
end
function Base.:≈(b1::Union{Bounds2, Bounds3}, b2::Union{Bounds2, Bounds3})
    b1.p_min ≈ b2.p_min && b1.p_max ≈ b2.p_max
end
function Base.getindex(b::Union{Bounds2, Bounds3}, i::Integer)
    i == 1 && return b.p_min
    i == 2 && return b.p_max
    error("Invalid index `$i`. Only `1` & `2` are valid.")
end

# Index through 8 corners.
function corner(b::Bounds3, c::Integer)
    c -= 1
    Point3f0(
        b[(c & 1) + 1][1],
        b[(c & 2) != 0 ? 2 : 1][2],
        b[(c & 4) != 0 ? 2 : 1][3],
    )
end

function Base.union(b1::Bounds3, b2::Bounds3)
    Bounds3(
        Point3f0(min.(b1.p_min, b2.p_min)),
        Point3f0(max.(b1.p_max, b2.p_max)),
    )
end

function Base.intersect(b1::Bounds3, b2::Bounds3)
    Bounds3(
        Point3f0(max.(b1.p_min, b2.p_min)),
        Point3f0(min.(b1.p_max, b2.p_max)),
    )
end

function overlaps(b1::Bounds3, b2::Bounds3)
    map((a, b) -> a && b, b1.p_max .>= b2.p_min, b1.p_min .<= b2.p_max) |> all
end

function inside(b::Bounds3, p::Point3f0)
    map((a, b) -> a && b, p .>= b.p_min, p .<= b.p_max) |> all
end

function inside_exclusive(b::Bounds3, p::Point3f0)
    map((a, b) -> a && b, p .>= b.p_min, p .< b.p_max) |> all
end

expand(b::Bounds3, δ::Float32) = Bounds3(b.p_min .- δ, b.p_max .+ δ)
diagonal(b::Bounds3) = b.p_max - b.p_min

function surface_area(b::Bounds3)
    d = b |> diagonal
    2 * (d[1] * d[2] + d[1] * d[3] + d[2] * d[3])
end

function volume(b::Bounds3)
    d = b |> diagonal
    d[1] * d[2] * d[3]
end

"""
Return index of the longest axis.
Useful for deciding which axis to subdivide,
when building ray-tracing acceleration structures.

1 - x, 2 - y, 3 - z.
"""
function maximum_extent(b::Bounds3)
    d = b |> diagonal
    if d[1] > d[2] && d[1] > d[3]
        return 1
    elseif d[2] > d[3]
        return 2
    end
    return 3
end

lerp(v1::Float32, v2::Float32, t::Float32) = (1 - t) * v1 + t * v2
lerp(p0::Point3f0, p1::Point3f0, t::Float32) = (1 - t) * p0 + t * p1
# Linearly interpolate point between the corners of the bounds.
lerp(b::Bounds3, p::Point3f0) = lerp.(p, b.p_min, b.p_max)

distance(p1::Point3f0, p2::Point3f0) = norm(p1 - p2)
function distance_squared(p1::Point3f0, p2::Point3f0)
    p = p1 - p2
    p ⋅ p
end

function offset(b::Bounds3, p::Point3f0)
    o = p - b.p_min
    g = b.p_max .> b.p_min
    !any(g) && return o
    o ./ Point3f0([
        gi ? b.p_max[i] - b.p_min[i] : 1f0
        for (i, gi) in enumerate(g)
    ])
end

function bounding_sphere(b::Bounds3)::Tuple{Point3f0, Float32}
    center = (b.p_min + b.p_max) / 2f0
    radius = inside(b, center) ? distance(center, b.p_max) : 0f0
    center, radius
end

const ϵ = eps(Float32) * 0.5f0
gamma(n::Float32)::Float32 = n * ϵ / (1 - n * ϵ)

function intersect_p(b::Bounds3, ray::Ray)::Tuple{Bool, Float32, Float32}
    t0, t1 = 0f0, ray.t_max
    for i in 1:3
        # Update interval for i-th bbox slab.
        inv_ray_dir = 1f0 / ray.d[i]
        t_near = (b.p_min[i] - ray.o[i]) * inv_ray_dir
        t_far = (b.p_max[i] - ray.o[i]) * inv_ray_dir
        # Swap if needed.
        if t_near > t_far
            t_near, t_far = t_far, t_near
        end
        # Update t_far to ensure robust ray-bounds intersection.
        t_far *= 1f0 + gamma(3f0)

        t0 = t_near > t0 ? t_near : t0
        t1 = t_far < t1 ? t_far : t1
        t0 > t1 && return false, 0f0, 0f0
    end
    true, t0, t1
end

is_dir_negative(dir::Vec3f0) = Point3{UInt8}([d < 0 ? 2 : 1 for d in dir])

"""
dir_is_negative: 1 -- false, 2 -- true
"""
function intersect_p(
    b::Bounds3, ray::Ray, inv_dir::Vec3f0, dir_is_negative::Point3{UInt8},
)
    tx_min = (b[dir_is_negative[1]][1] - ray.o[1]) * inv_dir[1]
    tx_max = (b[3 - dir_is_negative[1]][1] - ray.o[1]) * inv_dir[1]
    ty_min = (b[dir_is_negative[2]][2] - ray.o[2]) * inv_dir[2]
    ty_max = (b[3 - dir_is_negative[2]][2] - ray.o[2]) * inv_dir[2]
    (tx_min > tx_max || ty_min > ty_max) && return false
    if ty_min > tx_min
        tx_min = ty_min
    end
    if ty_max > tx_max
        tx_max = ty_max
    end

    tz_min = (b[dir_is_negative[3]][3] - ray.o[3]) * inv_dir[3]
    tz_max = (b[3 - dir_is_negative[3]][3] - ray.o[3]) * inv_dir[3]
    (tx_min > tz_max || tz_min > tx_max) && return false

    if tz_min > tx_min
        tx_min = tz_min
    end
    if tz_max < tx_max
        tx_max = tz_max
    end
    tx_min < ray.t_max && tx_max > 0
end
