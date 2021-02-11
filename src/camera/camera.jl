abstract type Camera end

struct CameraCore
    camera_to_world::Transformation
    shutter_open::Float32
    shutter_close::Float32
    film::Film
    # medium::Medium
end

struct CameraSample
    """
    Point on the film the ray passes through.
    """
    film::Point2f0
    """
    Point on the lens the ray passes through.
    """
    lens::Point2f0
    """
    Time at which the ray should sample the scene.
    Implementations should use this value to linearly interpolate between
    shutter_open & shutter_close time range.
    """
    time::Float32
end

"""
Compute the ray corresponding to a given sample.
It is IMPORTANT that the direction vector of ray is normalized.
Other parts of the system assume it to be so.

Returns generated ray & floating point that affects how much the radiance,
arriving at the film plane along generated ray, contributes to the final image.
Simple camera models can return 1, but cameras with simulated physical lenses
set this value to indicate how much light carries through the lenses,
based on their optical properties.
"""
function generate_ray(
    camera::C, sample::CameraSample,
)::Tuple{Ray, Float32} where C <: Camera
end
"""
Same as `generate_ray`, but also computes rays for pixels shifted one pixel
in x & y directions on the film plane.
Useful for anti-aliasing textures.
"""
function generate_ray_differential(
    camera::C, sample::CameraSample,
)::Tuple{RayDifferentials, Float32} where C <: Camera
    ray, wt = generate_ray(camera, sample)
    shifted_x = CameraSample(
        sample.film + Point2f0(1f0, 0f0), sample.lens, sample.time,
    )
    shifted_y = CameraSample(
        sample.film + Point2f0(0f0, 1f0), sample.lens, sample.time,
    )
    ray_x, wt_x = generate_ray(camera, shifted_x)
    ray_y, wt_y = generate_ray(camera, shifted_y)
    ray = RayDifferentials(
        ray.o, ray.d, ray.t_max, ray.time,
        true, ray_x.o, ray_y.o, ray_x.d, ray_y.d,
    )
    ray, wt
end

include("perspective.jl")
