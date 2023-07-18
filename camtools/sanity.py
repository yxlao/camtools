import numpy as np
import torch


def assert_numpy(x, name=None):
    if not isinstance(x, np.ndarray):
        maybe_name = f" {name}" if name is not None else ""
        raise ValueError(f"Expected{maybe_name} to be numpy array, but got {type(x)}.")


def assert_torch(x, name=None):
    if not torch.is_tensor(x):
        maybe_name = f" {name}" if name is not None else ""
        raise ValueError(f"Expected{maybe_name} to be torch tensor, but got {type(x)}.")


def assert_K(K):
    if K.shape != (3, 3):
        raise ValueError(f"K must has shape (3, 3), but got {K} of shape {K.shape}.")


def assert_T(T):
    if T.shape != (4, 4):
        raise ValueError(f"T must has shape (4, 4), but got {T} of shape {T.shape}.")
    if torch.is_tensor(T):
        is_valid = torch.allclose(
            T[3, :], torch.tensor([0, 0, 0, 1], dtype=T.dtype, device=T.device)
        )
    else:
        is_valid = np.allclose(T[3, :], np.array([0, 0, 0, 1]))
    if not is_valid:
        raise ValueError(f"T must has [0, 0, 0, 1] the bottom row, but got {T}.")


def assert_pose(pose):
    if pose.shape != (4, 4):
        raise ValueError(
            f"pose must has shape (4, 4), but got {pose} of shape {pose.shape}."
        )
    if torch.is_tensor(pose):
        is_valid = torch.allclose(
            pose[3, :], torch.tensor([0, 0, 0, 1], dtype=pose.dtype, device=pose.device)
        )
    else:
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


def assert_shape_nx3(x, name=None):
    assert_shape(x, (None, 3), name=name)


def assert_shape_nx2(x, name=None):
    assert_shape(x, (None, 2), name=name)


def assert_shape_4x4(x, name=None):
    assert_shape(x, (4, 4), name=name)


def assert_shape_3x4(x, name=None):
    assert_shape(x, (3, 4), name=name)


def assert_shape_3(x, name=None):
    assert_shape(x, (3,), name=name)


def assert_same_device(*tensors):
    """
    Args:
        tensors: list of tensors
    """
    if not isinstance(tensors, tuple):
        raise ValueError(f"Unknown input type: {type(tensors)}.")
    if len(tensors) == 0:
        return
    if len(tensors) == 1:
        if torch.is_tensor(tensors[0]) or isinstance(tensors[0], np.ndarray):
            return
        else:
            raise ValueError(f"Unknown input type: {type(tensors)}.")

    all_are_torch = all(torch.is_tensor(t) for t in tensors)
    all_are_numpy = all(isinstance(t, np.ndarray) for t in tensors)

    if not all_are_torch and not all_are_numpy:
        raise ValueError(f"All tensors must be torch tensors or numpy arrays.")

    if all_are_torch:
        devices = [t.device for t in tensors]
        if not all(devices[0] == d for d in devices):
            raise ValueError(
                f"All tensors must be on the same device, bui got {devices}."
            )
