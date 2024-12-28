import cv2
import numpy as np
import open3d as o3d
from jaxtyping import Float
from typing import Optional, Tuple

from . import sanity
from . import convert


def pad_0001(array):
    """
    Pad [0, 0, 0, 1] to the bottom row.

    Args:
        array: (3, 4) or (N, 3, 4).

    Returns:
        Array of shape (4, 4) or (N, 4, 4).
    """
    if array.ndim == 2:
        if not array.shape == (3, 4):
            raise ValueError(f"Expected array of shape (3, 4), but got {array.shape}.")
    elif array.ndim == 3:
        if not array.shape[-2:] == (3, 4):
            raise ValueError(
                f"Expected array of shape (N, 3, 4), but got {array.shape}."
            )
    else:
        raise ValueError(
            f"Expected array of shape (3, 4) or (N, 3, 4), but got {array.shape}."
        )

    if array.ndim == 2:
        bottom = np.array([0, 0, 0, 1], dtype=array.dtype)
        return np.concatenate([array, bottom[None, :]], axis=0)
    elif array.ndim == 3:
        bottom_single = np.array([0, 0, 0, 1], dtype=array.dtype)
        bottom = np.broadcast_to(bottom_single, (array.shape[0], 1, 4))
        return np.concatenate([array, bottom], axis=-2)
    else:
        raise ValueError("Should not reach here.")


def rm_pad_0001(array, check_vals=False):
    """
    Remove the bottom row of [0, 0, 0, 1].

    Args:
        array: (4, 4) or (N, 4, 4).
        check_vals (bool): If True, check that the bottom row is [0, 0, 0, 1].

    Returns:
        Array of shape (3, 4) or (N, 3, 4).
    """
    # Check shapes.
    if array.ndim == 2:
        if not array.shape == (4, 4):
            raise ValueError(f"Expected array of shape (4, 4), but got {array.shape}.")
    elif array.ndim == 3:
        if not array.shape[-2:] == (4, 4):
            raise ValueError(
                f"Expected array of shape (N, 4, 4), but got {array.shape}."
            )
    else:
        raise ValueError(
            f"Expected array of shape (4, 4) or (N, 4, 4), but got {array.shape}."
        )

    # Check vals.
    if check_vals:
        if array.ndim == 2:
            bottom = array[3, :]
            if not np.allclose(bottom, [0, 0, 0, 1]):
                raise ValueError(
                    f"Expected bottom row to be [0, 0, 0, 1], but got {bottom}."
                )
        elif array.ndim == 3:
            bottom = array[:, 3:4, :]
            expected_bottom = np.broadcast_to([0, 0, 0, 1], (array.shape[0], 1, 4))
            if not np.allclose(bottom, expected_bottom):
                raise ValueError(
                    f"Expected bottom row to be {expected_bottom}, but got {bottom}."
                )
        else:
            raise ValueError("Should not reach here.")

    return array[..., :3, :]


def to_homo(array):
    """
    Convert a 2D array to homogeneous coordinates by appending a column of ones.

    Args:
        array: A 2D numpy array of shape (N, M).

    Returns:
        A numpy array of shape (N, M+1) with a column of ones appended.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError(f"Input must be a 2D numpy array, but got {array.shape}.")

    ones = np.ones((array.shape[0], 1), dtype=array.dtype)
    return np.hstack((array, ones))


def from_homo(array):
    """
    Convert an array from homogeneous to Cartesian coordinates by dividing by the
    last column and removing it.

    Args:
        array: A 2D numpy array of shape (N, M) in homogeneous coordinates.

    Returns:
        A numpy array of shape (N, M-1) in Cartesian coordinates.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError(f"Input must be a 2D numpy array, but got {array.shape}.")
    if array.shape[1] < 2:
        raise ValueError(
            f"Input array must have at least two columns for removing "
            f"homogeneous coordinate, but got shape {array.shape}."
        )

    return array[:, :-1] / array[:, -1, np.newaxis]


def R_to_quat(R):
    """
    Convert rotation matrix to quaternion.

    Args:
        R: Rotation matrix of shape (3, 3) or (N, 3, 3).

    Returns:
        Quaternion of shape (4,) or (N, 4). The quaternion is normalized.
    """
    # https://github.com/isl-org/StableViewSynthesis/tree/main/co
    R = R.reshape(-1, 3, 3)
    q = np.empty((R.shape[0], 4), dtype=R.dtype)
    q[:, 0] = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]))
    q[:, 1] = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    q[:, 2] = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    q[:, 3] = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))
    q[:, 1] *= 2 * (R[:, 2, 1] > R[:, 1, 2]) - 1
    q[:, 2] *= 2 * (R[:, 0, 2] > R[:, 2, 0]) - 1
    q[:, 3] *= 2 * (R[:, 1, 0] > R[:, 0, 1]) - 1
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.squeeze()


def T_to_C(T: Float[np.ndarray, "4 4"]) -> Float[np.ndarray, "3"]:
    """
    Convert T to camera center.
    """
    sanity.assert_T(T)
    R, t = T[:3, :3], T[:3, 3]
    return R_t_to_C(R, t)


def pose_to_C(pose: Float[np.ndarray, "4 4"]) -> Float[np.ndarray, "3"]:
    """
    Convert pose to camera center.
    """
    sanity.assert_pose(pose)
    C = pose[:3, 3]
    return C


def T_to_pose(T):
    """
    Convert T to pose.
    """
    sanity.assert_T(T)
    return np.linalg.inv(T)


def pose_to_T(pose):
    """
    Convert pose to T.
    """
    sanity.assert_T(pose)
    return np.linalg.inv(pose)


def T_opengl_to_opencv(T: Float[np.ndarray, "4 4"]) -> Float[np.ndarray, "4 4"]:
    """
    Convert T from OpenGL convention to OpenCV convention.

    - OpenCV
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
        - Used in: OpenCV, COLMAP, camtools default
    - OpenGL
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
        - Used in: OpenGL, Blender, Nerfstudio
          https://docs.nerf.studio/quickstart/data_conventions.html#coordinate-conventions
    """
    sanity.assert_T(T)
    # pose = T_to_pose(T)
    # pose = pose_opengl_to_opencv(pose)
    # T = pose_to_T(pose)
    T = np.copy(T)
    T[1:3, 0:4] *= -1
    T = T[:, [1, 0, 2, 3]]
    T[:, 2] *= -1
    return T


def T_opencv_to_opengl(T: Float[np.ndarray, "4 4"]) -> Float[np.ndarray, "4 4"]:
    """
    Convert T from OpenCV convention to OpenGL convention.

    - OpenCV
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
        - Used in: OpenCV, COLMAP, camtools default
    - OpenGL
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
        - Used in: OpenGL, Blender, Nerfstudio
          https://docs.nerf.studio/quickstart/data_conventions.html#coordinate-conventions
    """
    sanity.assert_T(T)
    # pose = T_to_pose(T)
    # pose = pose_opencv_to_opengl(pose)
    # T = pose_to_T(pose)
    T = np.copy(T)
    T[:, 2] *= -1
    T = T[:, [1, 0, 2, 3]]
    T[1:3, 0:4] *= -1
    return T


def pose_opengl_to_opencv(pose: Float[np.ndarray, "4 4"]) -> Float[np.ndarray, "4 4"]:
    """
    Convert pose from OpenGL convention to OpenCV convention.

    - OpenCV
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
        - Used in: OpenCV, COLMAP, camtools default
    - OpenGL
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
        - Used in: OpenGL, Blender, Nerfstudio
          https://docs.nerf.studio/quickstart/data_conventions.html#coordinate-conventions
    """
    sanity.assert_pose(pose)
    pose = np.copy(pose)
    pose[2, :] *= -1
    pose = pose[[1, 0, 2, 3], :]
    pose[0:3, 1:3] *= -1
    return pose


def pose_opencv_to_opengl(pose: Float[np.ndarray, "4 4"]) -> Float[np.ndarray, "4 4"]:
    """
    Convert pose from OpenCV convention to OpenGL convention.

    - OpenCV
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
        - Used in: OpenCV, COLMAP, camtools default
    - OpenGL
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
        - Used in: OpenGL, Blender, Nerfstudio
          https://docs.nerf.studio/quickstart/data_conventions.html#coordinate-conventions
    """
    sanity.assert_pose(pose)
    pose = np.copy(pose)
    pose[0:3, 1:3] *= -1
    pose = pose[[1, 0, 2, 3], :]
    pose[2, :] *= -1
    return pose


def R_t_to_C(
    R: Float[np.ndarray, "3 3"],
    t: Float[np.ndarray, "3"],
) -> Float[np.ndarray, "3"]:
    """
    Convert R, t to camera center
    """
    # Equivalently,
    # C = - R.T @ t
    # C = - np.linalg.inv(R) @ t
    # C = pose[:3, 3] = np.linalg.inv(R_t_to_T(R, t))[:3, 3]

    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()


def R_C_to_t(
    R: Float[np.ndarray, "3 3"],
    C: Float[np.ndarray, "3"],
) -> Float[np.ndarray, "3"]:
    """
    Convert rotation matrix and camera center to translation vector.

    Args:
        R: Rotation matrix of shape (3, 3) or (N, 3, 3).
        C: Camera center of shape (3,) or (N, 3).

    Returns:
        Translation vector of shape (3,) or (N, 3).
    """
    # https://github.com/isl-org/StableViewSynthesis/blob/main/data/create_custom_track.py
    C = C.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    t = -R @ C
    return t.squeeze()


def roll_pitch_yaw_to_R(
    roll: float,
    pitch: float,
    yaw: float,
) -> Float[np.ndarray, "3 3"]:
    """
    Convert roll, pitch, and yaw angles to a rotation matrix.

    Args:
        roll: Rotation around the x-axis in radians.
        pitch: Rotation around the y-axis in radians.
        yaw: Rotation around the z-axis in radians.

    Returns:
        Rotation matrix of shape (3, 3).
    """
    rx_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    ry_pitch = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    rz_yaw = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R = rz_yaw @ ry_pitch @ rx_roll
    return R


def R_t_to_T(
    R: Float[np.ndarray, "3 3"],
    t: Float[np.ndarray, "3"],
) -> Float[np.ndarray, "4 4"]:
    """
    Convert rotation matrix and translation vector to transformation matrix.

    Args:
        R: Rotation matrix of shape (3, 3).
        t: Translation vector of shape (3,).

    Returns:
        Transformation matrix of shape (4, 4).
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_to_R_t(
    T: Float[np.ndarray, "4 4"],
) -> Tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "3"]]:
    sanity.assert_T(T)
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def P_to_K_R_t(
    P: Float[np.ndarray, "3 4"],
) -> Tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "3 3"], Float[np.ndarray, "3"]]:
    """
    Decompose projection matrix into intrinsic matrix, rotation matrix, and translation vector.

    Args:
        P: Projection matrix of shape (3, 4).

    Returns:
        Tuple containing:
        - K: Intrinsic matrix of shape (3, 3)
        - R: Rotation matrix of shape (3, 3)
        - t: Translation vector of shape (3,)
    """
    (
        camera_matrix,
        rot_matrix,
        trans_vect,
        rot_matrix_x,
        rot_matrix_y,
        rot_matrix_z,
        euler_angles,
    ) = cv2.decomposeProjectionMatrix(P)

    K = camera_matrix
    K = K / K[2, 2]
    R = rot_matrix
    t = -rot_matrix @ (trans_vect[:3] / trans_vect[3])

    return K, R, t.squeeze()


def P_to_K_T(
    P: Float[np.ndarray, "3 4"],
) -> Tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "4 4"]]:
    """
    Decompose projection matrix into intrinsic matrix and transformation matrix.

    Args:
        P: Projection matrix of shape (3, 4).

    Returns:
        Tuple containing:
        - K: Intrinsic matrix of shape (3, 3)
        - T: Transformation matrix of shape (4, 4)
    """
    K, R, t = P_to_K_R_t(P)
    T = R_t_to_T(R, t)
    return K, T


def K_T_to_P(
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
) -> Float[np.ndarray, "3 4"]:
    """
    Compute projection matrix from intrinsic matrix and transformation matrix.

    Args:
        K: Intrinsic matrix of shape (3, 3).
        T: Transformation matrix of shape (4, 4).

    Returns:
        Projection matrix of shape (3, 4).
    """
    return K @ T[:3, :]


def K_R_t_to_P(
    K: Float[np.ndarray, "3 3"],
    R: Float[np.ndarray, "3 3"],
    t: Float[np.ndarray, "3"],
) -> Float[np.ndarray, "3 4"]:
    """
    Compute projection matrix from intrinsic matrix, rotation matrix, and translation vector.

    Args:
        K: Intrinsic matrix of shape (3, 3).
        R: Rotation matrix of shape (3, 3).
        t: Translation vector of shape (3,).

    Returns:
        Projection matrix of shape (3, 4).
    """
    T = R_t_to_T(R, t)
    P = K @ T[:3, :]
    return P


def K_R_t_to_W2P(
    K: Float[np.ndarray, "3 3"],
    R: Float[np.ndarray, "3 3"],
    t: Float[np.ndarray, "3"],
) -> Float[np.ndarray, "4 4"]:
    """
    Compute world-to-projection matrix from intrinsic matrix, rotation matrix, and translation vector.

    Args:
        K: Intrinsic matrix of shape (3, 3).
        R: Rotation matrix of shape (3, 3).
        t: Translation vector of shape (3,).

    Returns:
        World-to-projection matrix of shape (4, 4).
    """
    return P_to_W2P(K_R_t_to_P(K, R, t))


def K_T_to_W2P(
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
) -> Float[np.ndarray, "4 4"]:
    """
    Compute world-to-projection matrix from intrinsic matrix and transformation matrix.

    Args:
        K: Intrinsic matrix of shape (3, 3).
        T: Transformation matrix of shape (4, 4).

    Returns:
        World-to-projection matrix of shape (4, 4).
    """
    R, t = T_to_R_t(T)
    return K_R_t_to_W2P(K, R, t)


def P_to_W2P(
    P: Float[np.ndarray, "3 4"],
) -> Float[np.ndarray, "4 4"]:
    """
    Convert projection matrix to world-to-projection matrix.

    Args:
        P: Projection matrix of shape (3, 4).

    Returns:
        World-to-projection matrix of shape (4, 4).
    """
    sanity.assert_shape_3x4(P, name="P")
    W2P = convert.pad_0001(P)
    return W2P


def W2P_to_P(
    W2P: Float[np.ndarray, "4 4"],
) -> Float[np.ndarray, "3 4"]:
    """
    Convert world-to-projection matrix to projection matrix.

    Args:
        W2P: World-to-projection matrix of shape (4, 4).

    Returns:
        Projection matrix of shape (3, 4).
    """
    if W2P.shape != (4, 4):
        raise ValueError(f"Expected W2P of shape (4, 4), but got {W2P.shape}.")
    P = convert.rm_pad_0001(W2P, check_vals=True)
    return P


def fx_fy_cx_cy_to_K(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Float[np.ndarray, "3 3"]:
    """
    Create intrinsic matrix from focal lengths and principal point coordinates.

    Args:
        fx: Focal length in x direction.
        fy: Focal length in y direction.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.

    Returns:
        Intrinsic matrix of shape (3, 3).
    """
    K = np.zeros((3, 3))
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1
    return K


def K_to_fx_fy_cx_cy(
    K: Float[np.ndarray, "3 3"],
) -> Tuple[float, float, float, float]:
    """
    Extract focal lengths and principal point coordinates from intrinsic matrix.

    Args:
        K: Intrinsic matrix of shape (3, 3).

    Returns:
        Tuple containing:
        - fx: Focal length in x direction
        - fy: Focal length in y direction
        - cx: Principal point x coordinate
        - cy: Principal point y coordinate
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    # <class 'numpy.float64'> to <class 'float'>
    return float(fx), float(fy), float(cx), float(cy)


def euler_to_R(
    yaw: float,
    pitch: float,
    roll: float,
) -> Float[np.ndarray, "3 3"]:
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


def spherical_to_T_towards_origin(
    radius: float,
    theta: float,
    phi: float,
) -> Float[np.ndarray, "4 4"]:
    """
    Convert spherical coordinates (ISO convention) to T, where the cameras looks
    at the origin from a distance (radius), and the camera up direction alines
    with the z-axis (the angle between the up direction and the z-axis is
    smaller than pi/2).

    Args:
        radius (float): Distance from the origin.
        theta (float): Inclination, angle w.r.t. positive polar (+z) axis.
            Range: [0, pi].
        phi (float): Azimuth, rotation angle from the initial meridian (x-y)
            plane. Range: [0, 2*pi].

    Returns:
        T of shape (4, 4).

    Ref:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system
    """
    if not 0 <= theta <= np.pi:
        raise ValueError(f"Expected theta in [0, pi], but got {theta}.")
    if not 0 <= phi <= 2 * np.pi:
        raise ValueError(f"Expected phi in [0, 2*pi], but got {phi}.")

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Default   : look at +Z, up is -Y.
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
    T = pose_to_T(pose)

    return T


def mesh_to_lineset(
    mesh: o3d.geometry.TriangleMesh,
    color: Optional[Float[np.ndarray, "3"]] = None,
) -> o3d.geometry.LineSet:
    """
    Convert Open3D mesh to Open3D lineset.
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise ValueError(f"Expected Open3D mesh, but got {type(mesh)}.")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    edges = set()
    for triangle in triangles:
        edges.add(tuple(sorted([triangle[0], triangle[1]])))
        edges.add(tuple(sorted([triangle[1], triangle[2]])))
        edges.add(tuple(sorted([triangle[2], triangle[0]])))

    edges = np.array(list(edges))

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(edges)

    if color is not None:
        if len(color) != 3:
            raise ValueError(f"Expected color of shape (3,), but got {color.shape}.")
        lineset.paint_uniform_color(color)

    return lineset


def im_distance_to_im_depth(
    im_distance: Float[np.ndarray, "h w"],
    K: Float[np.ndarray, "3 3"],
) -> Float[np.ndarray, "h w"]:
    """
    Convert distance image to depth image.

    Args:
        im_distance: Distance image (H, W), float.
        K: Camera intrinsic matrix (3, 3).

    Returns:
        Depth image (H, W), float.
    """
    if not im_distance.ndim == 2:
        raise ValueError(
            f"Expected im_distance of shape (H, W), but got {im_distance.shape}."
        )
    sanity.assert_K(K)
    height, width = im_distance.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dtype = im_distance.dtype

    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)

    u_norm = (u_grid - cx) / fx
    v_norm = (v_grid - cy) / fy
    norm_square = u_norm**2 + v_norm**2
    im_depth = im_distance / np.sqrt(norm_square + 1)
    im_depth = im_depth.astype(dtype)

    return im_depth


def im_depth_to_im_distance(
    im_depth: Float[np.ndarray, "h w"],
    K: Float[np.ndarray, "3 3"],
) -> Float[np.ndarray, "h w"]:
    """
    Convert depth image to distance image.

    Args:
        im_depth: Depth image (H, W), float.
        K: Camera intrinsic matrix (3, 3).

    Returns:
        Distance image (H, W), float.
    """
    if not im_depth.ndim == 2:
        raise ValueError(
            f"Expected im_depth of shape (H, W), but got {im_depth.shape}."
        )
    sanity.assert_K(K)
    height, width = im_depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dtype = im_depth.dtype

    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)

    u_norm = (u_grid - cx) / fx
    v_norm = (v_grid - cy) / fy
    norm_square = u_norm**2 + v_norm**2
    im_distance = im_depth * np.sqrt(norm_square + 1)
    im_distance = im_distance.astype(dtype)

    return im_distance
