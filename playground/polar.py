import camtools as ct
import numpy as np
import open3d as o3d


def euler_to_R(yaw, pitch, roll):
    R = np.array(
        [
            [
                np.cos(yaw) * np.cos(pitch),
                np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
                np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll),
            ],
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
                np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll),
            ],
            [
                -np.sin(pitch),
                np.cos(pitch) * np.sin(roll),
                np.cos(pitch) * np.cos(roll),
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


def polar_to_pose_towards_origin(r, theta, phi):
    """
    Convert polar coordinates (ISO convention) to pose pointing towards the origin.

    Args:
        r (float): Radius, distance from the origin.
        theta (float): Inclination, angle in the x-y plane.
        phi (float): Azimuth, angle from the z-axis.

    Returns:
        Pose of shape (4, 4).
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    yaw = np.pi / 2 - phi
    pitch = np.pi / 2 - theta
    roll = 0.0

    R = euler_to_R(yaw, pitch, roll)
    R = R.T

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]

    return pose


def main():
    # Create default camera intrinsics.
    K = np.array(
        [
            [554.256, 0.0, 320],
            [0.0, 554.256, 240],
            [0.0, 0.0, 1.0],
        ]
    )

    x, y, z = 0.0, 0.0, 0.0
    R = np.eye(3)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [x, y, z]
    T = ct.convert.pose_to_T(pose)

    Ks = [K]
    Ts = [T]

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    camera = ct.camera.create_camera_ray_frames(
        Ks=Ks, Ts=Ts, size=1.0, disable_up_triangle=True
    )
    o3d.visualization.draw_geometries([camera, axes])

    # # Create camera at x-y plane in a circle around the origin.
    # num_cams = 12
    # radius = 1.0
    # thetas = np.linspace(0, 2 * np.pi, num_cams, endpoint=False)

    # poses = [polar_to_pose_towards_origin(radius, theta, 0) for theta in thetas]

    # Ts = [ct.convert.pose_to_T(pose) for pose in poses]
    # Ks = [K for _ in range(num_cams)]

    # cameras = ct.camera.create_camera_ray_frames(Ks=Ks, Ts=Ts)
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    # o3d.visualization.draw_geometries([cameras, axes])


if __name__ == "__main__":
    main()
