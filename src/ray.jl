struct _Ray <: AbstractRay
    o::Point3f
    d::Vec3f
    t_max::Float32
    time::Float32
end

const Ray = MutableRef{_Ray}

function default(pool::MemoryPool, ::Type{Ray}; o::Point3f, d::Vec3f, t_max::Float32=Inf32, time::Float32=0.0f0)
    allocate(pool, Ray, (o, d, t_max, time))
end

struct _RayDifferentials <: AbstractRay
    o::Point3f
    d::Vec3f
    t_max::Float32
    time::Float32

    has_differentials::Bool
    rx_origin::Point3f
    ry_origin::Point3f
    rx_direction::Vec3f
    ry_direction::Vec3f
end
const RayDifferentials = MutableRef{_RayDifferentials}

function default(pool::MemoryPool, ::Type{RayDifferentials}; o::Point3f, d::Vec3f,
        t_max::Float32 = Inf32, time::Float32 = 0f0, has_differentials::Bool = false,
        rx_origin::Point3f = zeros(Point3f),
        ry_origin::Point3f = zeros(Point3f),
        rx_direction::Vec3f = zeros(Vec3f), ry_direction::Vec3f = zeros(Vec3f)
    )
    allocate(pool, RayDifferentials, (o, d, t_max, time, has_differentials, rx_origin, ry_origin, rx_direction, ry_direction))
end

@inline function allocate(pool::MemoryPool, ::Type{RayDifferentials}, r::Ray)::RayDifferentials
    default(pool, RayDifferentials, o = r.o, d = r.d, t_max = r.t_max, time = r.time)
end

@inline function set_direction!(r::MutableRef{<:AbstractRay}, d::Vec3f)
    r.d = map(i-> i ≈ 0f0 ? 0f0 : i, d)
end

@inline check_direction!(r::MutableRef{<:AbstractRay}) = set_direction!(r, r.d)

function apply(r::Union{Ray,RayDifferentials}, t::Number)
    r.o + r.d * t
end

function scale_differentials!(rd::RayDifferentials, s::Float32)
    rd.rx_origin = rd.o + (rd.rx_origin - rd.o) * s
    rd.ry_origin = rd.o + (rd.ry_origin - rd.o) * s
    rd.rx_direction = rd.d + (rd.rx_direction - rd.d) * s
    rd.rx_direction = rd.d + (rd.ry_direction - rd.d) * s
end
