import cv2
import numpy as np
import torch

from . import sanity


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

    if torch.is_tensor(array):
        if array.ndim == 2:
            bottom = torch.tensor([0, 0, 0, 1], dtype=array.dtype, device=array.device)
            return torch.cat([array, bottom[None, :]], dim=0)
        elif array.ndim == 3:
            bottom_single = torch.tensor(
                [0, 0, 0, 1], dtype=array.dtype, device=array.device
            )
            bottom = bottom_single[None, None, :].expand(array.shape[0], 1, 4)
            return torch.cat([array, bottom], dim=-2)
    else:
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
    Remove the homogeneous bottom row [0, 0, 0, 1].

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
        if torch.is_tensor(array):
            if array.ndim == 2:
                bottom = array[3, :]
                if not torch.allclose(
                    bottom, torch.tensor([0, 0, 0, 1], dtype=array.dtype)
                ):
                    raise ValueError(
                        f"Expected bottom row to be [0, 0, 0, 1], but got {bottom}."
                    )
            elif array.ndim == 3:
                bottom = array[:, 3:4, :]
                expected_bottom = torch.tensor([0, 0, 0, 1], dtype=array.dtype).expand(
                    array.shape[0], 1, 4
                )
                if not torch.allclose(bottom, expected_bottom):
                    raise ValueError(
                        f"Expected bottom row to be {expected_bottom}, but got {bottom}."
                    )
            else:
                raise ValueError("Should not reach here.")
        else:
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


def R_to_quat(R):
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


def T_opengl_to_opencv(T_opengl):
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
    """
    sanity.assert_T(T_opengl)

    R_b2p = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    R_opengl, t_opengl = T_to_R_t(T_opengl)
    R = R_b2p @ R_opengl
    t = t_opengl @ R_b2p
    T = R_t_to_T(R, t)

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
    """
    sanity.assert_T(T)

    R_b2p = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R_p2b = R_b2p.T

    R, t = T_to_R_t(T)
    R_opengl = R_p2b @ R
    t_opengl = t @ R_p2b
    T_opengl = R_t_to_T(R_opengl, t_opengl)

    return T_opengl


def pose_opengl_to_opencv(pose_opengl):
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
    """
    sanity.assert_pose(pose_opengl)
    pose = np.copy(pose_opengl)
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
    """
    sanity.assert_pose(pose)
    pose_opengl = np.copy(pose)
    pose_opengl[0:3, 1:3] *= -1
    return pose_opengl


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
    sanity.assert_same_device(R, t)
    if torch.is_tensor(R):
        T = torch.eye(4, device=R.device, dtype=R.dtype)
    else:
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
    if torch.is_tensor(P):
        bottom_row = torch.tensor([0, 0, 0, 1], device=P.device, dtype=P.dtype)
        W2P = torch.vstack((P, bottom_row))
    else:
        bottom_row = np.array([[0, 0, 0, 1]])
        W2P = np.vstack((P, bottom_row))
    return W2P


def W2P_to_P(W2P):
    P = W2P[:3, :4]
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
