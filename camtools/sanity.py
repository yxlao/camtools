import numpy as np


def assert_K(K):
    if K.shape != (3, 3):
        raise ValueError(f"K must has shape (3, 3), but got {K} of shape {K.shape}.")


def assert_T(T):
    if T.shape != (4, 4):
        raise ValueError(f"T must has shape (4, 4), but got {T} of shape {T.shape}.")
    is_valid = np.allclose(T[3, :], np.array([0, 0, 0, 1]))
    if not is_valid:
        raise ValueError(f"T must has [0, 0, 0, 1] the bottom row, but got {T}.")


def assert_pose(pose):
    if pose.shape != (4, 4):
        raise ValueError(
            f"pose must has shape (4, 4), but got {pose} of shape {pose.shape}."
        )
    is_valid = np.allclose(pose[3, :], np.array([0, 0, 0, 1]))
    if not is_valid:
        raise ValueError(f"pose must has [0, 0, 0, 1] the bottom row, but got {pose}.")


def assert_shape(x, shape, name=None):
    shape_valid = True

    if shape_valid and x.ndim != len(shape):
        shape_valid = False

    if shape_valid:
        for i, s in enumerate(shape):
            if s is not None:
                if x.shape[i] != s:
                    shape_valid = False
                    break

    if not shape_valid:
        name_must = f"{name} must" if name is not None else "Must"
        raise ValueError(f"{name_must} has shape {shape}, but got shape {x.shape}.")


def assert_shape_ndim(x, ndim, name=None):
    """
    Assert that x is a tensor with nd dimensions.

    Args:
        x: tensor
        nd: number of dimensions
        name: name of tensor
    """
    if x.ndim != ndim:
        name_must = f"{name} must" if name is not None else "Must"
        raise ValueError(f"{name_must} have {ndim} dimensions, but got {x.ndim}.")


def assert_shape_nx3(x, name=None):
    assert_shape(x, (None, 3), name=name)


def assert_shape_nx2(x, name=None):
    assert_shape(x, (None, 2), name=name)


def assert_shape_4x4(x, name=None):
    assert_shape(x, (4, 4), name=name)


def assert_shape_3x4(x, name=None):
    assert_shape(x, (3, 4), name=name)


def assert_shape_3x3(x, name=None):
    assert_shape(x, (3, 3), name=name)


def assert_shape_3(x, name=None):
    assert_shape(x, (3,), name=name)
