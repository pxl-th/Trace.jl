using LinearAlgebra: ×, ⋅

# In the previous version only doin this was necessary:
# GeometryBasics.@fixed_vector Normal StaticVector
# const Normal3f = Normal{3, Float32}


struct Normal3f
    vector::Vec3f
end

# Constructors
Normal3f(x::Real, y::Real, z::Real) = Normal3f(Vec3f(Float32(x), Float32(y), Float32(z)))
Normal3f(v::Real) = Normal3f(Vec3f(Float32(v), Float32(v), Float32(v)))
Normal3f() = Normal3f(0)

# Conversion from Normal3f to Vec3f
Base.convert(::Type{Vec3f}, n::Normal3f) = n.vector
Base.convert(::Type{Normal3f}, v::Vec3f) = Normal3f(v)

# Forward indexing operations
Base.getindex(n::Normal3f, i::Int) = getindex(n.vector, i)
Base.setindex!(n::Normal3f, v, i::Int) = setindex!(n.vector, v, i)
Base.iterate(n::Normal3f) = Base.iterate(n.vector)
Base.iterate(n::Normal3f, state) = Base.iterate(n.vector, state)

# Forward all other operations to Vec3f, wrapping the result in Normal3f if it's a Vec3f
for op in (:+, :-, :*, :/, :\, :^)
    @eval Base.$op(a::Normal3f, b) = Normal3f($op(a.vector, b))
    @eval Base.$op(a, b::Normal3f) = Normal3f($op(a, b.vector))
    @eval Base.$op(a::Normal3f, b::Normal3f) = Normal3f($op(a.vector, b.vector))
end

for op in (:+, :-)
    @eval Base.$op(a::Normal3f) = Normal3f($op(a.vector))
end

for op in (:×, :⋅)
    @eval LinearAlgebra.$op(a::Normal3f, b::Vec3f) = $op(a.vector, b)
    @eval LinearAlgebra.$op(a::Vec3f, b::Normal3f) = $op(a, b.vector)
    @eval LinearAlgebra.$op(a::Normal3f, b::Normal3f) = $op(a.vector, b.vector)
end

# Ensure that scalar multiplication returns a Normal3f
Base.:*(n::Normal3f, s::Real) = Normal3f(n.vector * s)
Base.:*(s::Real, n::Normal3f) = Normal3f(s * n.vector)

Base.length(n::Normal3f) = length(n.vector)

# Define ≈ for Normal3f by delegating to Vec3f
Base.:≈(a::Normal3f, b::Normal3f) = a.vector ≈ b.vector
# Optionally, allow comparison with Vec3f directly
Base.:≈(a::Normal3f, b::Vec3f) = a.vector ≈ b
Base.:≈(a::Vec3f, b::Normal3f) = a ≈ b.vector
# Forward norm function
LinearAlgebra.norm(n::Normal3f) = norm(n.vector)
# Show method to display as Normal3f
Base.show(io::IO, n::Normal3f) = print(io, "Normal3f(", n.vector, ")")
