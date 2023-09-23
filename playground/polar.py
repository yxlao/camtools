import camtools as ct
import numpy as np
import open3d as o3d
import itertools


def euler_to_R(yaw, pitch, roll):
    """
    Convert Euler angles to rotation matrix. Given a unit vector x, R @ x is x
    rotated by applying yaw, pitch, and roll consecutively. Ref:
    https://en.wikipedia.org/wiki/Euler_angles

    Args:
        yaw (float): Rotation around the z-axis (from x-axis to y-axis).
        pitch (float): Rotation around the y-axis (from z-axis to x-axis).
        roll (float): Rotation around the x-axis (from y-axis to z-axis).

    Returns:
        Rotation matrix R of shape (3, 3).
    """
    sin_y = np.sin(yaw)
    cos_y = np.cos(yaw)
    sin_p = np.sin(pitch)
    cos_p = np.cos(pitch)
    sin_r = np.sin(roll)
    cos_r = np.cos(roll)
    R = np.array(
        [
            [
                cos_y * cos_p,
                cos_y * sin_p * sin_r - sin_y * cos_r,
                cos_y * sin_p * cos_r + sin_y * sin_r,
            ],
            [
                sin_y * cos_p,
                sin_y * sin_p * sin_r + cos_y * cos_r,
                sin_y * sin_p * cos_r - cos_y * sin_r,
            ],
            [
                -sin_p,
                cos_p * sin_r,
                cos_p * cos_r,
            ],
        ]
    )
    return R


def polar_to_T_towards_origin(radius, theta, phi):
    """
    Convert polar coordinates (ISO convention) to T, where the cameras looks at
    the origin from a distance (radius), and the camera up direction alines with
    the z-axis (the angle between the up direction and the z-axis is smaller
    than pi/2.

    Args:
        radius (float): Distance from the origin.
        theta (float): Inclination, angle w.r.t. positive polar (+z) axis.
            Range: [0, pi].
        phi (float): Azimuth, rotation angle from the initial meridian (x-y)
            plane. Range: [0, 2*pi].

    Returns:
        T of shape (4, 4).
    """
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Before    : look at +Z, up is -Y.
    # After init: look at +X, up is +Z.
    init_R = euler_to_R(-np.pi / 2, 0, -np.pi / 2)
    # Rotate along z axis.
    phi_R = euler_to_R(phi + np.pi, 0, 0)
    # Rotate along y axis.
    theta_R = euler_to_R(0, np.pi / 2 - theta, 0)

    # Combine rotations, the order matters.
    R = phi_R @ theta_R @ init_R
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    T = ct.convert.pose_to_T(pose)

    return T


def main():
    """
    TODO: convert this to a unit test and check the center ray.
    """

    # Create default camera intrinsics.
    K = np.array(
        [
            [554.256, 0.0, 320],
            [0.0, 554.256, 240],
            [0.0, 0.0, 1.0],
        ]
    )

    # Create camera at x-y plane in a circle around the origin.
    resolution = 12
    radius = 1.0
    thetas = np.linspace(0, np.pi, resolution, endpoint=False)[1:]
    phis = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    thetas_phis = list(itertools.product(thetas, phis))

    Ts = [polar_to_T_towards_origin(radius, theta, phi) for (theta, phi) in thetas_phis]
    Ks = [K for _ in range(len(Ts))]

    cameras = ct.camera.create_camera_ray_frames(Ks=Ks, Ts=Ts, center_line=False)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([cameras, axes])


if __name__ == "__main__":
    main()
