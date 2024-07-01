import typing
from functools import wraps
from inspect import signature
from typing import Any, Tuple, Union

import jaxtyping
import numpy as np
import torch
from jaxtyping import _array_types

from . import backend


class Tensor:
    """
    An abstract tensor type for type hinting only. Typically np.ndarray or
    torch.Tensor is supported.
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
        valid_array_types = (np.ndarray, torch.Tensor)
    else:
        valid_array_types = (np.ndarray,)
    if not isinstance(arg, valid_array_types):
        raise TypeError(
            f"{arg_name} must be a tensor of type {valid_array_types}, "
            f"but got type {type(arg)}."
        )

    # Check shapes.
    gt_shape = _shape_from_dims_str(hint.dims)
    if not all(
        arg_dim == gt_dim or gt_dim is None
        for arg_dim, gt_dim in zip(arg.shape, gt_shape)
    ):
        raise TypeError(
            f"{arg_name} must be a tensor of shape {gt_shape}, "
            f"but got shape {arg.shape}."
        )

    # Check dtype.
    gt_dtypes = hint.dtypes
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
