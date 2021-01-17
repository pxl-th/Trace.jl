"""
Lambertian Reflection models a perfect diffuse surface
that scatters incident illumination equally in all directions.
"""
struct LambertianReflection{S <: Spectrum} <: BxDF
    """
    Reflectance spectrum, which is the fraction
    of incident light that is scattered.
    """
    r::S
    type::UInt8

    function LambertianReflection(r::S) where S <: Spectrum
        new{S}(r, BSDF_DIFFUSE | BSDF_REFLECTION)
    end
end

"""
Reflection distribution is constant and divides reflectance spectrum
equally over the hemisphere.
"""
function (l::LambertianReflection{S})(::Vec3f0, ::Vec3f0) where S <: Spectrum
    l.r * (1f0 / π)
end

"""
Directional-hemisphirical reflectance value is constant.
"""
function ρ(
    l::LambertianReflection{S}, ::Vec3f0, ::Int32, ::Vector{Point2f0},
) where S <: Spectrum
    l.r
end

"""
Hemispherical-hemisphirical reflectance value is constant.
"""
function ρ(
    l::LambertianReflection{S}, ::Vector{Point2f0}, ::Vector{Point2f0},
) where S <: Spectrum
    l.r
end


"""
Lambertian Transmission models perfect transmission.
"""
struct LambertianTransmission{S <: Spectrum} <: BxDF
    t::S
    type::UInt8

    function LambertianTransmission(t::S) where S <: Spectrum
        new{S}(t, BSDF_DIFFUSE | BSDF_TRANSMISSION)
    end
end

function (t::LambertianTransmission{S})(::Vec3f0, ::Vec3f0) where S <: Spectrum
    t.t * (1f0 / π)
end

function ρ(
    t::LambertianTransmission{S}, ::Vec3f0, ::Int32, ::Vector{Point2f0},
) where S <: Spectrum
    t.t
end

function ρ(
    t::LambertianTransmission{S}, ::Vector{Point2f0}, ::Vector{Point2f0},
) where S <: Spectrum
    t.t
end

function sample_f(
    b::LambertianTransmission{S}, wo::Vec3f0, sample::Point2f0,
) where S <: Spectrum
    wi = sample |> cosine_sample_hemisphere
    # Flipping the direction if necessary.
    wo[3] > 0 && (wi = Vec3f0(wi[1], wi[2], -wi[3]);)
    pdf = compute_pdf(b, wo, wi)
    wi, pdf, b(wo, wi), nothing
end

@inline function compute_pdf(
    ::LambertianTransmission, wo::Vec3f0, wi::Vec3f0,
)::Float32
    !same_hemisphere(wo, wi) ? abs(cos_θ(wi)) * (1f0 / π) : 0f0
end
