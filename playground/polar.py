import camtools as ct
import numpy as np
import open3d as o3d
import itertools

np.set_printoptions(precision=2, suppress=True)


def euler_to_R(yaw, pitch, roll):
    """
    Convert Euler angles to rotation matrix. Given a unit vector x, R @ x is x rotated by applying yaw, pitch, and roll consecutively.
    Ref: https://en.wikipedia.org/wiki/Euler_angles

    Args:
        yaw (float): Rotation around the z-axis (from x-axis to y-axis).
        pitch (float): Rotation around the y-axis (from z-axis to x-axis).
        roll (float): Rotation around the x-axis (from y-axis to z-axis).

    Returns:
        Rotation matrix R of shape (3, 3).
    """
    y, p, r = yaw, pitch, roll
    R = np.array(
        [
            [
                np.cos(y) * np.cos(p),
                np.cos(y) * np.sin(p) * np.sin(r) - np.sin(y) * np.cos(r),
                np.cos(y) * np.sin(p) * np.cos(r) + np.sin(y) * np.sin(r),
            ],
            [
                np.sin(y) * np.cos(p),
                np.sin(y) * np.sin(p) * np.sin(r) + np.cos(y) * np.cos(r),
                np.sin(y) * np.sin(p) * np.cos(r) - np.cos(y) * np.sin(r),
            ],
            [
                -np.sin(p),
                np.cos(p) * np.sin(r),
                np.cos(p) * np.cos(r),
            ],
        ]
    )

    return R


def two_unit_vectors_to_R(v1, v2):
    """
    Find rotation matrix R such that R @ v1 = v2.

    Args:
        v1 (np.ndarray): Vector of shape (3,). v1 will be normalized.
        v2 (np.ndarray): Vector of shape (3,). v2 will be normalized.

    Returns:
        Rotation matrix R of shape (3, 3).


    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # GG = @(A,B) [ dot(A,B) -norm(cross(A,B)) 0;\
    #             norm(cross(A,B)) dot(A,B)  0;\
    #             0              0           1];
    # FFi = @(A,B) [ A (B-dot(A,B)*A)/norm(B-dot(A,B)*A) cross(B,A) ];
    # UU = @(Fi,G) Fi*G*inv(Fi);

    GG = np.array(
        [
            [np.dot(v1, v2), -np.linalg.norm(np.cross(v1, v2)), 0],
            [np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2), 0],
            [0, 0, 1],
        ]
    )
    FFi = np.array(
        [
            [
                v1,
                (v2 - np.dot(v1, v2) * v1) / np.linalg.norm(v2 - np.dot(v1, v2) * v1),
                np.cross(v2, v1),
            ],
        ]
    )
    UU = FFi @ GG @ np.linalg.inv(FFi)

    return UU


def polar_to_T_towards_origin(radius, theta, phi):
    """
    Convert polar coordinates (ISO convention) to T, where the cameras looks at
    the origin from a distance (radius), and the camera up direction alines with
    the z-axis (the angle between the up direction and the z-axis is smaller
    than pi/2.

    Args:
        r (float): Radius, distance from the origin.
        theta (float): Inclination, angle w.r.t. positive polar (+z) axis.
            Range: [0, pi].
        phi (float): Azimuth, rotation angle from the initial meridian (x-y) plane.
            Range: [0, 2*pi].

    Returns:
        T of shape (4, 4).
    """

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    print(
        f"radius={radius:.2f}, theta={theta:.2f}, phi={phi:.2f}, "
        f"x={x:.2f}, y={y:.2f}, z={z:.2f}"
    )

    # Before    : look at +Z, up is -Y.
    # After init: look at -X, up is +Z.
    init_R = euler_to_R(np.pi / 2, 0, -np.pi / 2)
    theta_R = np.eye(3)
    phi_R = np.eye(3)

    # Rotating along z axis for phi degrees.
    phi_R = euler_to_R(0, 0, 0)
    # phi_R = euler_to_R(phi, 0, 0)

    # Rotating along y axis for theta degrees.
    theta_R = euler_to_R(0, 0, 0)
    # theta_R = euler_to_R(0, -(np.pi / 2 - theta), 0)

    R = theta_R @ phi_R @ init_R
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    T = ct.convert.pose_to_T(pose)

    return T


def main():
    # Create default camera intrinsics.
    K = np.array(
        [
            [554.256, 0.0, 320],
            [0.0, 554.256, 240],
            [0.0, 0.0, 1.0],
        ]
    )

    # Ks = [K]
    # Ts = [T]

    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # camera = ct.camera.create_camera_ray_frames(Ks=Ks, Ts=Ts, size=2.0)
    # o3d.visualization.draw_geometries([camera, axes])

    # # Create camera at x-y plane in a circle around the origin.
    resolution = 12
    radius = 1.0
    thetas = np.linspace(0, np.pi, resolution, endpoint=False)
    thetas = thetas[5:6]
    phis = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    thetas_phis = list(itertools.product(thetas, phis))

    Ts = [polar_to_T_towards_origin(radius, theta, phi) for (theta, phi) in thetas_phis]
    Ks = [K for _ in range(len(Ts))]

    cameras = ct.camera.create_camera_ray_frames(Ks=Ks, Ts=Ts, center_line=False)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([cameras, axes])


if __name__ == "__main__":
    main()
