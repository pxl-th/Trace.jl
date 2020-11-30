struct PointLight{S <: Spectrum} <: Light
    """
    Since point lights represent singularities that only emit light
    from a single position, flag is set to `LightδPosition`.
    """
    flags::UInt8
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
function sample_li(
    p::PointLight{S}, ref::Interaction, u::Point2f0,
)::Tuple{S, Vec3f0, Float32, VisibilityTester} where S <: Spectrum
    wi = Vec3f0(p.position - ref.p) |> normalize
    pdf = 1f0
    visibility = VisibilityTester(
        ref, Interaction(p.position, ref.time, Vec3f0(0f0), Normal3f0(0f0)),
    )
    radiance = p.i / distance_squared(p.position, ref.p)
    radiance, wi, pdf, visibility
end

"""
The total power emitted by the light source
over the entire sphere of directions.
"""
@inline function power(p::PointLight{S})::S where S <: Spectrum
    4f0 * π * p.i
end
