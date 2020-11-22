@with_kw mutable struct Ray
    o::Point3f0
    d::Vec3f0
    t_max::Float32 = Inf32
    time::Float32 = 0f0
end

@with_kw mutable struct RayDifferentials
    o::Point3f0
    d::Vec3f0
    t_max::Float32 = Inf32
    time::Float32 = 0f0

    has_differentials::Bool = false
    rx_origin::Point3f0 = zeros(Point3f0)
    ry_origin::Point3f0 = zeros(Point3f0)
    rx_direction::Vec3f0 = zeros(Vec3f0)
    ry_direction::Vec3f0 = zeros(Vec3f0)
end

function (r::Union{Ray, RayDifferentials})(t::Number)
    r.o + r.d * t
end

function scale_differentials!(rd::RayDifferentials, s::Float32)
    rd.rx_origin = rd.r.o + (rd.rx_origin - rd.r.o) * s
    rd.ry_origin = rd.r.o + (rd.ry_origin - rd.r.o) * s
    rd.rx_direction = rd.r.d + (rd.rx_direction - rd.r.d) * s
    rd.rx_direction = rd.r.d + (rd.ry_direction - rd.r.d) * s
end
