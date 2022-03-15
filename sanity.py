import numpy as np


def check_K(K):
    if K.shape != (3, 3):
        raise ValueError(
            f"K must has shape (3, 3), but got {K} of shape {K.shape}.")


def check_T(T):
    if T.shape != (4, 4):
        raise ValueError(
            f"T must has shape (4, 4), but got {T} of shape {T.shape}.")
    if not np.allclose(T[3, :], [0, 0, 0, 1]):
        raise ValueError(
            f"T must has [0, 0, 0, 1] the bottom row, but got {T}.")


def check_shape_Nx3(x, name=None):
    if x.ndim != 2 or x.shape[1] != 3:
        name_must = f"{name} must" if name is not None else "Must"
        raise ValueError(
            f"{name_must} has shape (N, 3), but got shape {x.shape}.")


def check_shape_3(x, name=None):
    if x.shape != (3,):
        name_must = f"{name} must" if name is not None else "Must"
        raise ValueError(
            f"{name_must} has shape (3,), but got shape {x.shape}.")
