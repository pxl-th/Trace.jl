struct PointLight{S <: Spectrum} <: Light
    """
    Since point lights represent singularities that only emit light
    from a single position, flag is set to `LightδPosition`.
    """
    flags::LightFlags
    """
    Ligh-source is positioned at the origin of its light space.
    """
    light_to_world::Transformation
    world_to_light::Transformation

    i::S
    """
    Position in world space.
    """
    position::Point3f0

    function PointLight(light_to_world::Transformation, i::S) where S <: Spectrum
        new{S}(
            LightδPosition, light_to_world, light_to_world |> inv,
            i, Point3f0(0f0) |> light_to_world,
        )
    end
end

"""
Compute radiance arriving at `ref.p` interaction point at `ref.time` time
due to that light, assuming there are no occluding objects between them.

# Args

- `p::PointLight`: Light which illuminates the interaction point `ref`.
- `ref::Interaction`: Interaction point for which to compute radiance.
- `u::Point2f0`: Sampling point that is ignored for `PointLight`,
    since it has no area.

# Returns

`Tuple{S, Vec3f0, Float32, VisibilityTester} where S <: Spectrum`:

    - `S`: Computed radiance.
    - `Vec3f0`: Incident direction to the light source `wi`.
    - `Float32`: Probability density for the light sample that was taken.
        For `PointLight` it is always `1`.
    - `VisibilityTester`: Initialized visibility tester that holds the
        shadow ray that must be traced to verify that
        there are no occluding objects between the light and reference point.
"""
function sample_li(p::PointLight, ref::Interaction, ::Point2f0)
    wi = Vec3f0(p.position - ref.p) |> normalize
    pdf = 1f0
    visibility = VisibilityTester(
        ref, Interaction(p.position, ref.time, Vec3f0(0f0), Normal3f0(0f0)),
    )
    radiance = p.i / distance_squared(p.position, ref.p)
    radiance, wi, pdf, visibility
end

function sample_le(
    p::PointLight, u1::Point2f0, u2::Point2f0, time::Float32,
)::Tuple{RGBSpectrum, Ray, Normal3f0, Float32, Float32}
    ray = Ray(o=p.position, d=uniform_sample_sphere(u1))
    @assert norm(ray.d) ≈ 1f0
    light_normal = ray.d |> Normal3f0
    pdf_pos = 1f0
    pdf_dir = uniform_sphere_pdf()
    p.i, ray, light_normal, pdf_pos, pdf_dir
end

"""
Total power emitted by the light source over the entire sphere of directions.
"""
@inline function power(p::PointLight)
    4f0 * π * p.i
end
