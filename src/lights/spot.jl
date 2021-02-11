struct SpotLight{S <: Spectrum} <: Light
    flags::LightFlags
    light_to_world::Transformation
    world_to_light::Transformation
    position::Point3f0
    i::S
    cos_total_width::Float32
    cos_falloff_start::Float32

    function SpotLight(
        light_to_world::Transformation, i::S,
        total_width::Float32, falloff_start::Float32,
    ) where S <: Spectrum
        new{S}(
            LightδPosition, light_to_world, light_to_world |> inv,
            Point3f0(0f0) |> light_to_world, i,
            total_width |> deg2rad |> cos, falloff_start |> deg2rad |> cos,
        )
    end
end

function sample_li(s::SpotLight, ref::Interaction, ::Point2f0)
    wi = Vec3f0(s.position - ref.p) |> normalize
    pdf = 1f0
    visibility = VisibilityTester(
        ref, Interaction(s.position, ref.time, Vec3f0(0f0), Normal3f0(0f0)),
    )
    radiance = s.i * falloff(s, -wi) / distance_squared(s.position, ref.p)
    radiance, wi, pdf, visibility
end

function falloff(s::SpotLight, w::Vec3f0)::Float32
    wl = w |> s.world_to_light |> normalize
    cosθ = wl[3]
    cosθ < s.cos_total_width && return 0f0
    cosθ ≥ s.cos_falloff_start && return 1f0
    # Compute falloff inside spotlight cone.
    δ = (cosθ - s.cos_total_width) / (s.cos_falloff_start - s.cos_total_width)
    δ ^ 4
end

@inline function power(s::SpotLight)
    s.i * 2f0 * π * (1f0 - 0.5f0 * (s.cos_falloff_start + s.cos_total_width))
end

function sample_le(
    s::SpotLight, u1::Point2f0, u2::Point2f0, time::Float32,
)::Tuple{RGBSpectrum, Ray, Normal3f0, Float32, Float32}
    w = uniform_sample_cone(u1, s.cos_total_width) |> s.light_to_world
    ray = Ray(o=s.position, d=w)
    @assert norm(ray.d) ≈ 1f0
    light_normal = ray.d |> Normal3f0
    pdf_pos = 1f0
    pdf_dir = uniform_cone_pdf(s.cos_total_width)
    s.i * falloff(s, ray.d), ray, light_normal, pdf_pos, pdf_dir
end
