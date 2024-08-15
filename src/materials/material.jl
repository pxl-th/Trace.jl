MATTE_MATERIAL = UInt8(1)

function MatteMaterial(
        Kd::Texture, σ::Texture,
    )
    return UberMaterial(MATTE_MATERIAL; Kd=Kd, σ=σ)
end


"""
Compute scattering function.
"""
Base.Base.@propagate_inbounds  function matte_material(pool, m::UberMaterial, si::SurfaceInteraction, ::Bool, transport)
    # TODO perform bump mapping
    # Evaluate textures and create BSDF.
    bsdf = BSDF(pool, si)
    r = clamp(m.Kd(si))
    is_black(r) && return
    σ = clamp(m.σ(si), 0f0, 90f0)
    bsdfs = bsdf.bxdfs
    if σ ≈ 0f0
        bsdfs[1] = LambertianReflection(r)
    else
        bsdfs[1] = OrenNayar(r, σ)
    end
    bsdfs.last = 1
    return bsdf
end

const MIRROR_MATERIAL = UInt8(2)

function MirrorMaterial(Kr::Texture)
    return UberMaterial(MIRROR_MATERIAL; Kr=Kr)
end

Base.Base.@propagate_inbounds function mirror_material(pool, m::UberMaterial, si::SurfaceInteraction, ::Bool, transport)
    bsdf = BSDF(pool, si)
    r = clamp(m.Kr(si))
    is_black(r) && return
    bxdfs = bsdf.bxdfs
    bxdfs[1] = SpecularReflection(r, FresnelNoOp())
    bxdfs.last = 1
    return bsdf
end

const GLASS_MATERIAL = UInt8(3)

function GlassMaterial(
        Kr::Texture, Kt::Texture, u_roughness::Texture, v_roughness::Texture, index::Texture,
        remap_roughness::Bool,
    )
    return UberMaterial(GLASS_MATERIAL; Kr=Kr, Kt=Kt, u_roughness=u_roughness, v_roughness=v_roughness, index=index, remap_roughness=remap_roughness)
end

Base.Base.@propagate_inbounds function glass_material(pool, g::UberMaterial, si::SurfaceInteraction, allow_multiple_lobes::Bool, transport)

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
        bsdfs[1] = FresnelSpecular(r, t, 1.0f0, η, transport)
        bsdfs.last = 1
        return
    end

    if g.remap_roughness
        u_roughness = roughness_to_α(u_roughness)
        v_roughness = roughness_to_α(v_roughness)
    end
    distribution = is_specular ? nothing : TrowbridgeReitzDistribution(
        u_roughness, v_roughness,
    )
    last = 0
    if !is_black(r)
        fresnel = FresnelDielectric(1f0, η)
        if is_specular
            bsdfs[1] = SpecularReflection(r, fresnel)
        else
            bsdfs[1] = MicrofacetReflection(r, distribution, fresnel, transport)
        end
        last = 1
    end
    if !is_black(t)
        last += 1
        if is_specular
            bsdfs[last] = SpecularTransmission(t, 1.0f0, η, transport)
        else
            bsdfs[last] = MicrofacetTransmission(t, distribution, 1.0f0, η, transport)
        end
    end
    bsdfs.last = last
    return bsdf
end

const PLASTIC_MATERIAL = UInt8(4)

function PlasticMaterial(
        Kd::Texture, Ks::Texture, roughness::Texture, remap_roughness::Bool,
    )
    return UberMaterial(PLASTIC_MATERIAL; Kd=Kd, Ks=Ks, roughness=roughness, remap_roughness=remap_roughness)
end

Base.Base.@propagate_inbounds function plastic_material(pool, p::UberMaterial,
        si::SurfaceInteraction, ::Bool, transport,
    )
    bsdf = BSDF(pool, si)
    bsdfs = bsdf.bxdfs
    # Initialize diffuse componen of plastic material.
    kd = clamp(p.Kd(si))
    last = 0
    if !is_black(kd)
        bsdfs[1] = LambertianReflection(kd)
        last += 1
    end
    # Initialize specular component.
    ks = clamp(p.Ks(si))
    is_black(ks) && return
    # Create microfacet distribution for plastic material.
    fresnel = FresnelDielectric(1.5f0, 1f0)
    rough = p.roughness(si)
    p.remap_roughness && (rough = roughness_to_α(rough))
    distribution = TrowbridgeReitzDistribution(rough, rough)
    last += 1
    bsdfs[last] = MicrofacetReflection(ks, distribution, fresnel, transport)
    bsdfs.last = last
    return bsdf
end
