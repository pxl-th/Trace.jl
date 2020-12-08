"""
Lambertian Reflection models a perfect diffuse surface
that scatters incident illumination equally in all directions.
"""
struct LambertianReflection{S <: Spectrum}
    """
    Reflectance spectrum, which is the fraction
    of incident light that is scattered.
    """
    r::S
end

function Base.:&(::LambertianReflection, t::BxDFTypes)::Bool
    t & BSDF_DIFFUSE || t & BSDF_REFLECTION
end

"""
Reflection distribution is constant and divides reflectance spectrum
equally over the hemisphere.
"""
function (l::LambertianReflection{S})(::Vec3f0, ::Vec3f0)::S where S <: Spectrum
    l.r * (1f0 / π)
end

"""
Directional-hemisphirical reflectance value is constant.
"""
function ρ(
    l::LambertianReflection{S}, ::Vec3f0, ::Int32, ::Vector{Point2f0},
)::S where S <: Spectrum
    l.r
end

"""
Hemispherical-hemisphirical reflectance value is constant.
"""
function ρ(
    l::LambertianReflection{S}, ::Vector{Point2f0}, ::Vector{Point2f0},
)::S where S <: Spectrum
    l.r
end


"""
Lambertian Transmission models perfect transmission.
"""
struct LambertianTransmission{S <: Spectrum}
    t::S
end

function Base.:&(::LambertianTransmission, t::BxDFTypes)::Bool
    t & BSDF_DIFFUSE || t & BSDF_TRANSMISSION
end

function (t::LambertianTransmission{S})(::Vec3f0, ::Vec3f0) where S <: Spectrum
    t.t * (1f0 / π)
end

function ρ(
    t::LambertianTransmission{S}, ::Vec3f0, ::Int32, ::Vector{Point2f0},
)::S where S <: Spectrum
    t.t
end

function ρ(
    t::LambertianTransmission{S}, ::Vector{Point2f0}, ::Vector{Point2f0},
)::S where S <: Spectrum
    t.t
end

function sample_f(
    b::LambertianTransmission{S}, wo::Vec3f0, sample::Point2f0,
)::Tuple{Vec3f0, Float32, S} where {S <: Spectrum}
    wi = sample |> cosine_sample_hemisphere
    # Flipping the direction if necessary.
    wo[3] > 0 && (wi[3] *= -1)
    pdf = compute_pdf(b, wo, wi)
    wi, pdf, b(wo, wi)
end

@inline function compute_pdf(::LambertianTransmission, wo::Vec3f0, wi::Vec3f0)::Float32
    !same_hemisphere(wo, wi) ? abs(cos_θ(wi)) * (1f0 / π) : 0
end
