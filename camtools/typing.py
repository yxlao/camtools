import typing
from distutils.version import LooseVersion
from functools import wraps
from inspect import signature
from typing import Any, Tuple, Union

import jaxtyping
import numpy as np
import pkg_resources

from . import backend


class Tensor:
    """
    An abstract tensor type for type hinting only.
    Typically np.ndarray or torch.Tensor is supported.
    """

    pass


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

    if backend.is_torch_available():
        import torch

        if isinstance(dtype, torch.dtype):
            return str(dtype).split(".")[1]

    return ValueError(f"Unknown dtype {dtype}.")


def _shape_from_dim_str(dim_str: str) -> Tuple[Union[int, None, str], ...]:
    shape = []
    elements = dim_str.split()
    for elem in elements:
        if elem == "...":
            shape.append("...")
        elif elem.isdigit():
            shape.append(int(elem))
        else:
            shape.append(None)
    return tuple(shape)


def _is_shape_compatible(
    arg_shape: Tuple[Union[int, None, str], ...],
    gt_shape: Tuple[Union[int, None, str], ...],
) -> bool:
    if "..." in gt_shape:
        if len(arg_shape) < len(gt_shape) - 1:
            return False
        # We only support one ellipsis for now
        if gt_shape.count("...") > 1:
            raise ValueError(
                "Only one ellipsis is supported in the shape hint for now."
            )

        # Compare dimensions before and after the ellipsis
        pre_ellipsis = gt_shape.index("...")
        post_ellipsis = len(gt_shape) - pre_ellipsis - 1
        return all(
            arg_shape[i] == gt_shape[i] or gt_shape[i] is None
            for i in range(pre_ellipsis)
        ) and all(
            arg_shape[-i - 1] == gt_shape[-i - 1] or gt_shape[-i - 1] is None
            for i in range(post_ellipsis)
        )
    else:
        if len(arg_shape) != len(gt_shape):
            return False
        return all(
            arg_dim == gt_dim or gt_dim is None
            for arg_dim, gt_dim in zip(arg_shape, gt_shape)
        )


def _assert_tensor_hint(
    hint: jaxtyping.AbstractArray,
    arg: Any,
    arg_name: str,
):
    """
    Args:
        hint: A type hint for a tensor, must be javtyping.AbstractArray.
        arg: An argument to check, typically a tensor.
        arg_name: The name of the argument, for error messages.
    """
    # Check array types.
    if backend.is_torch_available():
        import torch

        valid_array_types = (np.ndarray, torch.Tensor)
    else:
        valid_array_types = (np.ndarray,)
    if not isinstance(arg, valid_array_types):
        raise TypeError(
            f"{arg_name} must be of type {valid_array_types}, "
            f"but got type {type(arg)}."
        )

    # Check shapes.
    gt_shape = _shape_from_dim_str(hint.dim_str)
    if not _is_shape_compatible(arg.shape, gt_shape):
        raise TypeError(
            f"{arg_name} must be of shape {gt_shape}, but got shape {arg.shape}."
        )

    # Check dtype.
    gt_dtypes = hint.dtypes
    if _dtype_to_str(arg.dtype) not in gt_dtypes:
        raise TypeError(
            f"{arg_name} must be of dtype {gt_dtypes}, "
            f"but got dtype {_dtype_to_str(arg.dtype)}."
        )


def check_shape_and_dtype(func):
    """
    A decorator to enforce type and shape specifications as per type hints.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        arg_name_to_arg = bound_args.arguments
        arg_name_to_hint = typing.get_type_hints(func)

        for arg_name, arg in arg_name_to_arg.items():
            if arg_name in arg_name_to_hint:
                hint = arg_name_to_hint[arg_name]
                if issubclass(hint, jaxtyping.AbstractArray):
                    _assert_tensor_hint(hint, arg, arg_name)

        return func(*args, **kwargs)

    return wrapper
