using GeometryBasics

mutable struct AtomicVec3f0
    x::Threads.Atomic{Float32}
    y::Threads.Atomic{Float32}
    z::Threads.Atomic{Float32}
end

function AtomicVec3f0()
    AtomicVec3f0(
        Threads.Atomic{Float32}(0f0),
        Threads.Atomic{Float32}(0f0),
        Threads.Atomic{Float32}(0f0),
    )
end

function set!(a::AtomicVec3f0, v::Float32)
    a.x[] = v
    a.y[] = v
    a.z[] = v
end
function set!(a::AtomicVec3f0, p::Point3f0)
    a.x[] = p[1]
    a.y[] = p[2]
    a.z[] = p[3]
end

function Threads.atomic_add!(a::AtomicVec3f0, p::Point3f0)
    Threads.atomic_add!(a.x, p[1])
    Threads.atomic_add!(a.y, p[2])
    Threads.atomic_add!(a.z, p[3])
end

function Base.convert(::Type{Point3f0}, a::AtomicVec3f0)
    Point3f0(a.x[], a.y[], a.z[])
end

function main()
    a = AtomicVec3f0()

    Threads.@threads for i in 1:100_000
        Threads.atomic_add!(a, Point3f0(1f0))
    end

    println(a)
    println(convert(Point3f0, a))
    set!(a, Point3f0(0f0))
    println(a)
end

main()
