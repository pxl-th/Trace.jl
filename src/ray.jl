Base.@kwdef struct Ray <: AbstractRay
    o::Point3f
    d::Vec3f
    t_max::Float32 = Inf32
    time::Float32 = 0.0f0
end

@inline function Ray(ray::Ray; o::Point3f = ray.o, d::Vec3f = ray.d, t_max::Float32 = ray.t_max, time::Float32 = ray.time)
    Ray(o, d, t_max, time)
end


Base.@kwdef struct RayDifferentials <: AbstractRay
    o::Point3f
    d::Vec3f
    t_max::Float32 = Inf32
    time::Float32 = 0.0f0

    has_differentials::Bool = false
    rx_origin::Point3f = zeros(Point3f)
    ry_origin::Point3f = zeros(Point3f)
    rx_direction::Vec3f = zeros(Vec3f)
    ry_direction::Vec3f = zeros(Vec3f)
end

@inline function RayDifferentials(ray::RayDifferentials;
        o::Point3f = ray.o, d::Vec3f = ray.d, t_max::Float32 = ray.t_max, time::Float32 = ray.time,
        has_differentials::Bool = ray.has_differentials, rx_origin::Point3f = ray.rx_origin, ry_origin::Point3f = ray.ry_origin,
        rx_direction::Vec3f = ray.rx_direction, ry_direction::Vec3f = ray.ry_direction
    )
    RayDifferentials(o, d, t_max, time, has_differentials, rx_origin, ry_origin, rx_direction, ry_direction)
end

@inline function RayDifferentials(r::Ray)::RayDifferentials
    RayDifferentials(o = r.o, d = r.d, t_max = r.t_max, time = r.time)
end

@inline function set_direction(r::Ray, d::Vec3f)
    d = map(i-> i ≈ 0f0 ? 0f0 : i, d)
    return Ray(r, d=d)
end

@inline function set_direction(r::RayDifferentials, d::Vec3f)
    d = map(i -> i ≈ 0.0f0 ? 0.0f0 : i, d)
    return RayDifferentials(r, d=d)
end

@inline check_direction(r::AbstractRay) = set_direction(r, r.d)

apply(r::AbstractRay, t::Number) = r.o + r.d * t

@inline function scale_differentials(rd::RayDifferentials, s::Float32)
    return RayDifferentials(rd;
        rx_origin = rd.o + (rd.rx_origin - rd.o) * s,
        ry_origin = rd.o + (rd.ry_origin - rd.o) * s,
        rx_direction = rd.d + (rd.rx_direction - rd.d) * s,
        ry_direction = rd.d + (rd.ry_direction - rd.d) * s
    )
end
