import cv2
import numpy as np

from . import sanity
from . import convert

from .backend import (
    Tensor,
    tensor_backend_numpy,
    tensor_backend_auto,
    is_torch_available,
    ivy,
    torch,
    create_array,
    create_ones,
    create_empty,
    get_tensor_backend,
)
from jaxtyping import Float
from typing import List, Tuple, Dict, Union
from matplotlib import pyplot as plt


@tensor_backend_auto
def pad_0001(
    array: Union[Float[Tensor, "3 4"], Float[Tensor, "N 3 4"]]
) -> Union[Float[Tensor, "4 4"], Float[Tensor, "N 4 4"]]:
    """
    Pad [0, 0, 0, 1] to the bottom row.

    Args:
        array: NumPy or Torch array of shape (3, 4) or (N, 3, 4).

    Returns:
        NumPy or Torch array of shape (4, 4) or (N, 4, 4).
    """
    dtype = array.dtype
    backend = get_tensor_backend(array)
    bottom = create_array([0, 0, 0, 1], dtype=dtype, backend=backend)

    if array.ndim == 2:
        return ivy.concat([array, ivy.expand_dims(bottom, axis=0)], axis=0)
    elif array.ndim == 3:
        bottom = ivy.expand_dims(bottom, axis=0)
        bottom = ivy.broadcast_to(bottom, (array.shape[0], 1, 4))
        return ivy.concat([array, bottom], axis=1)
    else:
        raise ValueError("Input array must be 2D or 3D.")


@tensor_backend_auto
def rm_pad_0001(
    array: Union[Float[Tensor, "4 4"], Float[Tensor, "N 4 4"]],
    check_vals: bool = False,
) -> Union[Float[Tensor, "3 4"], Float[Tensor, "N 3 4"]]:
    """
    Remove the bottom row of [0, 0, 0, 1].

    Args:
        array: (4, 4) or (N, 4, 4).
        check_vals (bool): If True, check that the bottom row is [0, 0, 0, 1].

    Returns:
        Array of shape (3, 4) or (N, 3, 4).
    """
    # Check vals.
    if check_vals:
        backend = get_tensor_backend(array)
        dtype = array.dtype
        gt_bottom = create_array([0, 0, 0, 1], dtype=dtype, backend=backend)

        if array.ndim == 2:
            bottom = array[3, :]
        elif array.ndim == 3:
            bottom = array[:, 3:4, :]
        else:
            raise ValueError(f"Invalid array shape {array.shape}.")

        if not ivy.allclose(bottom, gt_bottom):
            raise ValueError(
                f"Expected bottom row to be {gt_bottom}, but got {bottom}."
            )

    return array[..., :3, :]


@tensor_backend_auto
def to_homo(array: Float[Tensor, "n m"]) -> Float[Tensor, "n m+1"]:
    """
    Convert a 2D array to homogeneous coordinates by appending a column of ones.

    Args:
        array: A 2D numpy array of shape (N, M).

    Returns:
        A numpy array of shape (N, M+1) with a column of ones appended.
    """
    backend = get_tensor_backend(array)
    ones = create_ones((array.shape[0], 1), dtype=array.dtype, backend=backend)
    return ivy.concat([array, ones], axis=1)


@tensor_backend_auto
def from_homo(array: Float[Tensor, "n m"]) -> Float[Tensor, "n m-1"]:
    """
    Convert an array from homogeneous to Cartesian coordinates by dividing by the
    last column and removing it.

    Args:
        array: A 2D array of shape (N, M) in homogeneous coordinates, where M >= 2.

    Returns:
        An array of shape (N, M-1) in Cartesian coordinates.
    """
    if array.shape[1] < 2:
        raise ValueError(
            f"Input array must have at least two columns, "
            f"but got shape {array.shape}."
        )

    return array[:, :-1] / array[:, -1:]


@tensor_backend_auto
def R_to_quat(
    R: Union[Float[Tensor, "n 3 3"], Float[Tensor, "3 3"]]
) -> Union[Float[Tensor, "n 4"], Float[Tensor, "4"]]:
    """
    Convert a batch of rotation matrices or a single rotation matrix from
    rotation matrix form to quaternion form.

    Args:
        R: A tensor containing either a single (3, 3) rotation matrix or a batch
           of (n, 3, 3) rotation matrices.

    Returns:
        A tensor of quaternions. If the input is (3, 3), the output will be (4,),
        and if the input is (n, 3, 3), the output will be (n, 4).

    Ref:
        https://github.com/isl-org/StableViewSynthesis/tree/main/co
    """
    orig_shape = R.shape
    R = ivy.reshape(R, (-1, 3, 3))
    q = create_empty((R.shape[0], 4), dtype=R.dtype, backend=get_tensor_backend(R))
    q[:, 0] = ivy.sqrt(ivy.maximum(0, 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]))
    q[:, 1] = ivy.sqrt(ivy.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    q[:, 2] = ivy.sqrt(ivy.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    q[:, 3] = ivy.sqrt(ivy.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))
    q[:, 1] *= 2 * (R[:, 2, 1] > R[:, 1, 2]) - 1
    q[:, 2] *= 2 * (R[:, 0, 2] > R[:, 2, 0]) - 1
    q[:, 3] *= 2 * (R[:, 1, 0] > R[:, 0, 1]) - 1
    q = q / ivy.vector_norm(q, axis=1, keepdims=True)

    # Handle different input shapes for squeezing
    if orig_shape == (3, 3):
        return ivy.squeeze(q)
    else:
        return q


def T_to_C(T):
    """
    Convert T to camera center.
    """
    sanity.assert_T(T)
    R, t = T[:3, :3], T[:3, 3]
    return R_t_to_C(R, t)


def pose_to_C(pose):
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


def T_opengl_to_opencv(T):
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


def T_opencv_to_opengl(T):
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


def pose_opengl_to_opencv(pose):
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


def pose_opencv_to_opengl(pose):
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


def R_t_to_C(R, t):
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


def R_C_to_t(R, C):
    # https://github.com/isl-org/StableViewSynthesis/blob/main/data/create_custom_track.py
    C = C.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    t = -R @ C
    return t.squeeze()


def roll_pitch_yaw_to_R(roll, pitch, yaw):
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


def R_t_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_to_R_t(T):
    sanity.assert_T(T)
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def P_to_K_R_t(P):
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


def P_to_K_T(P):
    K, R, t = P_to_K_R_t(P)
    T = R_t_to_T(R, t)
    return K, T


def K_T_to_P(K, T):
    return K @ T[:3, :]


def K_R_t_to_P(K, R, t):
    T = R_t_to_T(R, t)
    P = K @ T[:3, :]
    return P


def K_R_t_to_W2P(K, R, t):
    return P_to_W2P(K_R_t_to_P(K, R, t))


def K_T_to_W2P(K, T):
    R, t = T_to_R_t(T)
    return K_R_t_to_W2P(K, R, t)


def P_to_W2P(P):
    sanity.assert_shape_3x4(P, name="P")
    W2P = convert.pad_0001(P)
    return W2P


def W2P_to_P(W2P):
    if W2P.shape != (4, 4):
        raise ValueError(f"Expected W2P of shape (4, 4), but got {W2P.shape}.")
    P = convert.rm_pad_0001(W2P, check_vals=True)
    return P


def fx_fy_cx_cy_to_K(fx, fy, cx, cy):
    K = np.zeros((3, 3))
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1
    return K


def K_to_fx_fy_cx_cy(K):
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
