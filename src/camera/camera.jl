abstract type Camera end

struct CameraCore
    camera_to_world::Transformation
    shutter_open::Float32
    shutter_close::Float32
end

# Does this camera need a per-ray time sample?  Only motion-blur cameras do
# (shutter_open != shutter_close).  Used by the volpath camera-ray kernel to
# skip a Sobol dimension when not needed — see compute_pixel_sample's 6-arg
# overload in sampler/sobol.jl.  Default true (conservative); concrete
# camera types override.
camera_uses_motion_blur(::Camera) = true

# Does this camera need a per-ray lens sample?  Only depth-of-field cameras
# do (lens_radius > 0).  Same skip-when-not-needed pattern as above.
camera_uses_lens(::Camera) = true

struct CameraSample
    """
    Point on the film the ray passes through.
    """
    film::Point2f
    """
    Point on the lens the ray passes through.
    """
    lens::Point2f
    """
    Time at which the ray should sample the scene.
    Implementations should use this value to linearly interpolate between
    shutter_open & shutter_close time range.
    """
    time::Float32
    """
    Filter weight for this sample. Used when accumulating samples into pixels.
    Computed from filter importance sampling - samples at the center of the
    filter have higher weight than those at the edges.
    """
    filter_weight::Float32
end

# Convenience constructor for backwards compatibility (weight defaults to 1.0)
CameraSample(film::Point2f, lens::Point2f, time::Float32) =
    CameraSample(film, lens, time, 1.0f0)

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
    )::Tuple{Ray,Float32} where C<:Camera
end

include("perspective.jl")
include("matrix.jl")

# Generic accessor for camera_to_world transform (handles different nesting depths)
# PerspectiveCamera: camera.core (ProjectiveCamera) .core (CameraCore) .camera_to_world
# MatrixCamera: camera.core (CameraCore) .camera_to_world
get_camera_to_world(camera::PerspectiveCamera) = camera.core.core.camera_to_world
get_camera_to_world(camera::MatrixCamera) = camera.core.camera_to_world

get_camera_position(camera::PerspectiveCamera) = Raycore.transform_point(get_camera_to_world(camera).m, Point3f(0f0))
get_camera_position(camera::MatrixCamera) = Raycore.transform_point(get_camera_to_world(camera).m, Point3f(0f0))
