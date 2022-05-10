import cv2
import numpy as np
import torch

from . import sanity


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
    R, t = T[:3, :3], T[:3, 3]
    return R_t_to_C(R, t)


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
    rx_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    ry_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
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
    (camera_matrix, rot_matrix, trans_vect, rot_matrix_x, rot_matrix_y,
     rot_matrix_z, euler_angles) = cv2.decomposeProjectionMatrix(P)

    K = camera_matrix
    R = rot_matrix
    t = -rot_matrix @ (trans_vect[:3] / trans_vect[3])

    return K, R, t.squeeze()


def P_to_K_T(P):
    K, R, t = P_to_K_R_t(P)
    T = R_t_to_T(R, t)
    return K, T


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
