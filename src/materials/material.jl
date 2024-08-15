MATTE_MATERIAL = UInt8(1)

function MatteMaterial(
        Kd::Texture, σ::Texture,
    )
    return UberMaterial(MATTE_MATERIAL; Kd=Kd, σ=σ)
end

"""
Compute scattering function.
"""
function matte_material(pool, m::UberMaterial, si::SurfaceInteraction, ::Bool, transport)
    # TODO perform bump mapping
    # Evaluate textures and create BSDF.
    bsdf = BSDF(pool, si)
    r = clamp(m.Kd(si))
    is_black(r) && return
    σ = clamp(m.σ(si), 0f0, 90f0)
    bsdfs = bsdf.bxdfs
    if σ ≈ 0f0
        push!(bsdfs, LambertianReflection(r))
    else
        push!(bsdfs, OrenNayar(r, σ))
    end
    return bsdf
end

const MIRROR_MATERIAL = UInt8(2)

function MirrorMaterial(Kr::Texture)
    return UberMaterial(MIRROR_MATERIAL; Kr=Kr)
end

function mirror_material(pool, m::UberMaterial, si::SurfaceInteraction, ::Bool, transport)
    bsdf = BSDF(pool, si)
    r = clamp(m.Kr(si))
    is_black(r) && return
    push!(bsdf.bxdfs, SpecularReflection(r, FresnelNoOp()))
    return bsdf
end

const GLASS_MATERIAL = UInt8(3)

function GlassMaterial(
        Kr::Texture, Kt::Texture, u_roughness::Texture, v_roughness::Texture, index::Texture,
        remap_roughness::Bool,
    )
    return UberMaterial(GLASS_MATERIAL; Kr=Kr, Kt=Kt, u_roughness=u_roughness, v_roughness=v_roughness, index=index, remap_roughness=remap_roughness)
end

function glass_material(pool, g::UberMaterial, si::SurfaceInteraction, allow_multiple_lobes::Bool, transport)

    η = g.index(si)
    u_roughness = g.u_roughness(si)
    v_roughness = g.v_roughness(si)

    bsdf = BSDF(pool, si, η)
    bsdfs = bsdf.bxdfs
    r = clamp(g.Kr(si))
    t = clamp(g.Kt(si))
    is_black(r) && is_black(t) && return

    is_specular = u_roughness ≈ 0 && v_roughness ≈ 0
    if is_specular && allow_multiple_lobes
        push!(bsdfs, FresnelSpecular(r, t, 1.0f0, η, transport))
        return
    end

    if g.remap_roughness
        u_roughness = roughness_to_α(u_roughness)
        v_roughness = roughness_to_α(v_roughness)
    end
    distribution = is_specular ? nothing : TrowbridgeReitzDistribution(
        u_roughness, v_roughness,
    )

    if !is_black(r)
        fresnel = FresnelDielectric(1f0, η)
        if is_specular
            push!(bsdfs, SpecularReflection(r, fresnel))
        else
            push!(bsdfs, MicrofacetReflection(r, distribution, fresnel, transport))
        end
    end
    if !is_black(t)
        if is_specular
            push!(bsdfs, SpecularTransmission(t, 1.0f0, η, transport))
        else
            push!(bsdfs, MicrofacetTransmission(t, distribution, 1.0f0, η, transport))
        end
    end
    return bsdf
end

const PLASTIC_MATERIAL = UInt8(4)

function PlasticMaterial(
        Kd::Texture, Ks::Texture, roughness::Texture, remap_roughness::Bool,
    )
    return UberMaterial(PLASTIC_MATERIAL; Kd=Kd, Ks=Ks, roughness=roughness, remap_roughness=remap_roughness)
end

function plastic_material(pool, p::UberMaterial,
        si::SurfaceInteraction, ::Bool, transport,
    )

    bsdf = BSDF(pool, si)
    bsdfs = bsdf.bxdfs
    # Initialize diffuse componen of plastic material.
    kd = clamp(p.Kd(si))
    !is_black(kd) && push!(bsdfs, LambertianReflection(kd))
    # Initialize specular component.
    ks = clamp(p.Ks(si))
    is_black(ks) && return
    # Create microfacet distribution for plastic material.
    fresnel = FresnelDielectric(1.5f0, 1f0)
    rough = p.roughness(si)
    p.remap_roughness && (rough = roughness_to_α(rough))
    distribution = TrowbridgeReitzDistribution(rough, rough)
    push!(bsdfs, MicrofacetReflection(ks, distribution, fresnel, transport))
    return bsdf
end
