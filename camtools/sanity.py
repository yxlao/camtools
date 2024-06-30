import numpy as np
import numpy as np
import torch
import typing

from functools import wraps
from jaxtyping import _array_types
from typing import Tuple, Union

from typing import Union, Tuple, get_args
import jaxtyping


def _dtype_to_str(dtype):
    """
    Convert numpy or torch dtype to string

    - "bool"
    - "bool_"
    - "uint4"
    - "uint8"
    - "uint16"
    - "uint32"
    - "uint64"
    - "int4"
    - "int8"
    - "int16"
    - "int32"
    - "int64"
    - "bfloat16"
    - "float16"
    - "float32"
    - "float64"
    - "complex64"
    - "complex128"
    """
    if isinstance(dtype, np.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        return str(dtype).split(".")[1]
    else:
        raise ValueError(f"Unknown dtype {dtype}.")


def _shape_from_dims_str(
    dims: Tuple[Union[_array_types._FixedDim, _array_types._NamedDim], ...]
) -> Tuple[Union[int, None], ...]:
    shape = []
    for dim in dims:
        if isinstance(dim, _array_types._FixedDim):
            shape.append(dim.size)
        elif isinstance(dim, _array_types._NamedDim):
            shape.append(None)
    return tuple(shape)


def _assert_tensor_hint(hint, arg, arg_name):
    # Unpack Union types.
    if getattr(hint, "__origin__", None) is Union:
        unpacked_hints = get_args(hint)
    else:
        unpacked_hints = (hint,)

    # If there exists one non jaxtyping hint, skip the check.
    for unpacked_hint in unpacked_hints:
        if not issubclass(unpacked_hint, jaxtyping.AbstractArray):
            return

    # Check array types (e.g. np.ndarray, torch.Tensor, ...)
    valid_array_types = tuple(
        unpacked_hint.array_type for unpacked_hint in unpacked_hints
    )
    if not isinstance(arg, valid_array_types):
        raise TypeError(
            f"{arg_name} must be a tensor of type {valid_array_types}, "
            f"but got type {type(arg)}."
        )

    # Check shapes.
    gt_shape = _shape_from_dims_str(unpacked_hints[0].dims)
    for unpacked_hint in unpacked_hints:
        if _shape_from_dims_str(unpacked_hint.dims) != gt_shape:
            raise TypeError(
                f"Internal error: all shapes in the Union must be the same, "
                f"but got {gt_shape} and {_shape_from_dims_str(unpacked_hint.dims)}."
            )
    if not all(
        arg_dim == gt_dim or gt_dim is None
        for arg_dim, gt_dim in zip(arg.shape, gt_shape)
    ):
        raise TypeError(
            f"{arg_name} must be a tensor of shape {gt_shape}, "
            f"but got shape {arg.shape}."
        )

    # Check dtype.
    gt_dtypes = unpacked_hints[0].dtypes  # A tuple of dtype names (str)
    for unpacked_hint in unpacked_hints:
        if unpacked_hint.dtypes != gt_dtypes:
            raise TypeError(
                f"Internal error: all dtypes in the Union must be the same, "
                f"but got {gt_dtypes} and {unpacked_hint.dtypes}."
            )
    if _dtype_to_str(arg.dtype) not in gt_dtypes:
        raise TypeError(
            f"{arg_name} must be a tensor of dtype {gt_dtypes}, "
            f"but got dtype {_dtype_to_str(arg.dtype)}."
        )


def check_shape_and_dtype(func):
    """
    A decorator to enforce type and shape specifications as per type hints.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = typing.get_type_hints(func)
        all_args = {
            **dict(zip(func.__code__.co_varnames[: func.__code__.co_argcount], args)),
            **kwargs,
        }

        for arg_name, arg in all_args.items():
            if arg_name in hints:
                hint = hints[arg_name]
                _assert_tensor_hint(hint, arg, arg_name)

        return func(*args, **kwargs)

    return wrapper


def assert_numpy(x, name=None):
    if not isinstance(x, np.ndarray):
        maybe_name = f" {name}" if name is not None else ""
        raise ValueError(f"Expected{maybe_name} to be numpy array, but got {type(x)}.")


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
