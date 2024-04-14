import cv2
import numpy as np

from . import sanity
from . import convert

from typing import Tuple, Optional, Union


def pad_0001(array: np.ndarray) -> np.ndarray:
    """
    Pad [0, 0, 0, 1] to the bottom row of the input array.

    Args:
        array: A numpy array of shape (3, 4) or (N, 3, 4).

    Returns:
        A numpy array with the shape (4, 4) if input was (3, 4) or (N, 4, 4) if input was (N, 3, 4).
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


def rm_pad_0001(array: np.ndarray, check_vals: bool = False) -> np.ndarray:
    """
    Remove the bottom row [0, 0, 0, 1] from the input array.

    Args:
        array: A numpy array of shape (4, 4) or (N, 4, 4).
        check_vals: If True, verifies the bottom row is [0, 0, 0, 1] before removing.

    Returns:
        A numpy array with the shape (3, 4) if input was (4, 4) or (N, 3, 4) if input was (N, 4, 4).
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


def to_homo(array: np.ndarray) -> np.ndarray:
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


def from_homo(array: np.ndarray) -> np.ndarray:
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


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix R to a quaternion representation.

    Args:
        R: A numpy array containing one or more rotation matrices, shaped (N, 3, 3).

    Returns:
        A numpy array containing the quaternion(s), shaped (N, 4).
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


def T_to_C(T: np.ndarray) -> np.ndarray:
    """
    Convert the extrinsic matrix T to the camera center C.

    Args:
        T: A 4x4 extrinsic matrix.

    Returns:
        The camera center C as a numpy array.
    """
    sanity.assert_T(T)
    R, t = T[:3, :3], T[:3, 3]
    return R_t_to_C(R, t)


def pose_to_C(pose: np.ndarray) -> np.ndarray:
    """
    Convert the extrinsic matrix T to the camera pose.

    Args:
        T: A 4x4 extrinsic matrix.

    Returns:
        The camera pose as a numpy array.
    """
    sanity.assert_pose(pose)
    C = pose[:3, 3]
    return C


def T_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert the camera pose to the extrinsic matrix T.

    Args:
        C: A 3x1 camera pose.

    Returns:
        The extrinsic matrix T as a numpy array.
    """
    sanity.assert_T(T)
    return np.linalg.inv(T)


def pose_to_T(pose: np.ndarray) -> np.ndarray:
    """
    Convert the camera pose to the extrinsic matrix T.

    Args:
        pose: A 4x4 camera pose.

    Returns:
        The extrinsic matrix T as a numpy array.
    """
    sanity.assert_T(pose)
    return np.linalg.inv(pose)


def T_opengl_to_opencv(T: np.ndarray) -> np.ndarray:
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

    Args:
        T: The 4x4 extrinsic matrix in OpenGL convention.

    Returns:
        The 4x4 extrinsic matrix in OpenCV convention.
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


def T_opencv_to_opengl(T: np.ndarray) -> np.ndarray:
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

    Args:
        T: The 4x4 extrinsic matrix T in OpenCV convention.

    Returns:
        The 4x4 extrinsic matrix T in OpenGL convention.
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


def pose_opengl_to_opencv(pose: np.ndarray) -> np.ndarray:
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

    Args:
        pose: A 4x4 pose matrix in OpenGL convention.

    Returns:
        The pose matrix converted to OpenCV convention.
    """
    sanity.assert_pose(pose)
    pose = np.copy(pose)
    pose[2, :] *= -1
    pose = pose[[1, 0, 2, 3], :]
    pose[0:3, 1:3] *= -1
    return pose


def pose_opencv_to_opengl(pose: np.ndarray) -> np.ndarray:
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

    Args:
        pose: A 4x4 pose matrix in OpenCV convention.

    Returns:
        The pose matrix converted to OpenGL convention.
    """
    sanity.assert_pose(pose)
    pose = np.copy(pose)
    pose[0:3, 1:3] *= -1
    pose = pose[[1, 0, 2, 3], :]
    pose[2, :] *= -1
    return pose


def R_t_to_C(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert rotation (R) and translation (t) matrices to a camera center.
    The camera center is computed as -R.T * t, representing the inverse of the
    transformation from world to camera coordinates.

    Args:
        R: A 3x3 rotation matrix.
        t: A 3-element translation vector.

    Returns:
        An array of shape (3,) representing the camera center.
    """
    # Equivalently,
    # C = - R.T @ t
    # C = - np.linalg.inv(R) @ t
    # C = pose[:3, 3] = np.linalg.inv(R_t_to_T(R, t))[:3, 3]

    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()


def R_C_to_t(R, C):
    """
    Convert rotation matrix (R) and camera center (C) to a translation vector (t).
    The translation vector is computed considering the camera center and rotation,
    typically used for reconstructing the full extrinsic matrix.

    Args:
        R: A 3x3 rotation matrix.
        C: Camera center, of shape (3,).

    Returns:
        An array of shape (3,) representing the translation vector t.
    """
    # https://github.com/isl-org/StableViewSynthesis/blob/main/data/create_custom_track.py
    C = C.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    t = -R @ C
    return t.squeeze()


def roll_pitch_yaw_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll, pitch, and yaw angles to a rotation matrix. The angles are
    applied in the order of yaw (around z-axis), pitch (around y-axis), and
    roll (around x-axis).

    Args:
        roll: Rotation around the x-axis in radians.
        pitch: Rotation around the y-axis in radians.
        yaw: Rotation around the z-axis in radians.

    Returns:
        A 3x3 rotation matrix.
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


def R_t_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Combine a rotation matrix (R) and a translation vector (t) into a full
    extrinsic matrix (T), which transforms points from world coordinates to
    camera coordinates.

    Args:
        R: A 3x3 rotation matrix.
        t: A 3-element translation vector.

    Returns:
        A 4x4 transformation matrix.
    """
    sanity.assert_same_device(R, t)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_to_R_t(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a transformation matrix (T) into its rotation (R) and translation (t)
    components.

    Args:
        T: A 4x4 transformation matrix.

    Returns:
        A tuple containing a 3x3 rotation matrix and a 3-element translation vector.
    """
    sanity.assert_T(T)
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def P_to_K_R_t(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a camera projection matrix (P) into intrinsic matrix (K), rotation
    matrix (R), and translation vector (t).

    Args:
        P: A 3x4 camera projection matrix.

    Returns:
        A tuple of intrinsic matrix (3x3), rotation matrix (3x3), and translation
        vector (3,).
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


def P_to_K_T(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a camera projection matrix (P) into intrinsic matrix (K) and a full
    extrinsic matrix (T).

    Args:
        P: A 3x4 camera projection matrix.

    Returns:
        A tuple of the intrinsic matrix (3x3) and the full extrinsic matrix (4x4).
    """
    K, R, t = P_to_K_R_t(P)
    T = R_t_to_T(R, t)
    return K, T


def K_T_to_P(K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Construct a camera projection matrix (P) from an intrinsic matrix (K) and an
    extrinsic matrix (T).

    Args:
        K: A 3x3 intrinsic matrix.
        T: A 4x4 extrinsic matrix.

    Returns:
        A 3x4 camera projection matrix.
    """
    return K @ T[:3, :]


def K_R_t_to_P(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construct a camera projection matrix (P) from intrinsic matrix (K), rotation
    matrix (R), and translation vector (t).

    Args:
        K: A 3x3 intrinsic matrix.
        R: A 3x3 rotation matrix.
        t: A 3-element translation vector.

    Returns:
        A 3x4 camera projection matrix.
    """
    T = R_t_to_T(R, t)
    P = K @ T[:3, :]
    return P


def K_R_t_to_W2P(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construct a full camera projection matrix (W2P) in homogeneous coordinates
    from intrinsic matrix (K), rotation matrix (R), and translation vector (t).

    Args:
        K: A 3x3 intrinsic matrix.
        R: A 3x3 rotation matrix.
        t: A 3-element translation vector.

    Returns:
        A 4x4 camera projection matrix in homogeneous coordinates.
    """
    return P_to_W2P(K_R_t_to_P(K, R, t))


def K_T_to_W2P(K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Convert an intrinsic matrix (K) and a transformation matrix (T) to a full
    camera projection matrix (W2P) in homogeneous coordinates.

    Args:
        K: A 3x3 intrinsic matrix.
        T: A 4x4 transformation matrix.

    Returns:
        A 4x4 camera projection matrix in homogeneous coordinates.
    """
    R, t = T_to_R_t(T)
    return K_R_t_to_W2P(K, R, t)


def P_to_W2P(P: np.ndarray) -> np.ndarray:
    """
    Convert a 3x4 camera projection matrix (P) to a 4x4 homogeneous projection matrix (W2P).

    Args:
        P: A 3x4 camera projection matrix.

    Returns:
        A 4x4 homogeneous projection matrix.
    """
    sanity.assert_shape_3x4(P, name="P")
    W2P = convert.pad_0001(P)
    return W2P


def W2P_to_P(W2P: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 homogeneous projection matrix (W2P) back to a 3x4 camera projection matrix (P).

    Args:
        W2P: A 4x4 homogeneous projection matrix.

    Returns:
        A 3x4 camera projection matrix.
    """
    if W2P.shape != (4, 4):
        raise ValueError(f"Expected W2P of shape (4, 4), but got {W2P.shape}.")
    P = convert.rm_pad_0001(W2P, check_vals=True)
    return P


def fx_fy_cx_cy_to_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Construct an intrinsic matrix (K) from focal lengths (fx, fy) and principal point (cx, cy).

    Args:
        fx: Focal length along the x-axis.
        fy: Focal length along the y-axis.
        cx: x-coordinate of the principal point.
        cy: y-coordinate of the principal point.

    Returns:
        A 3x3 intrinsic matrix.
    """
    K = np.zeros((3, 3))
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1
    return K


def K_to_fx_fy_cx_cy(K: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Extract focal lengths and principal point coordinates from an intrinsic matrix (K).

    Args:
        K: A 3x3 intrinsic matrix.

    Returns:
        A tuple containing focal lengths (fx, fy) and principal point coordinates (cx, cy).
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    # <class 'numpy.float64'> to <class 'float'>
    return float(fx), float(fy), float(cx), float(cy)


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


def spherical_to_T_towards_origin(radius, theta, phi):
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
