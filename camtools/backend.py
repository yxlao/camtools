import inspect
import typing
import warnings
from functools import lru_cache, wraps
from typing import Any, Literal, Tuple, Union, Dict, List

import jaxtyping
import numpy as np


@lru_cache(maxsize=1)
def _safely_import_torch():
    """
    Open3D has an issue where it must be imported before torch. If Open3D is
    installed, this function will import Open3D before torch. Otherwise, it
    will return simply import and return torch.

    Use this function to import torch within camtools to handle the Open3D
    import order issue. That is, within camtools, we shall avoid `import torch`,
    and instead use `from camtools.backend import torch`. As torch is an
    optional dependency for camtools, this function will return None if torch
    is not available.

    Returns:
        module: The torch module if available, otherwise None.
    """
    try:
        __import__("open3d")
    except ImportError:
        pass

    try:
        _torch = __import__("torch")
        return _torch
    except ImportError:
        return None


torch = _safely_import_torch()


@lru_cache(maxsize=1)
def is_torch_available():
    return _safely_import_torch() is not None


@lru_cache(maxsize=1)
def _safely_import_ivy():
    """
    This function sets up the warnings filter to suppress the deprecation
    before importing ivy. This is a temporary workaround to suppress the
    deprecation warning from numpy 2.0.

    Within camtools, we shall avoid `import ivy`, and instead use
    `from camtools.backend import ivy`.
    """
    warnings.filterwarnings(
        "ignore",
        message=".*numpy.core.numeric is deprecated.*",
        category=DeprecationWarning,
        module="ivy",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Compositional function.*array_mode is set to False.*",
        category=UserWarning,
        module="ivy",
    )
    ivy = __import__("ivy")
    ivy.set_array_mode(False)
    return ivy


ivy = _safely_import_ivy()


@lru_cache(maxsize=64)
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

    if is_torch_available():
        if isinstance(dtype, torch.dtype):
            return str(dtype).split(".")[1]

    return ValueError(f"Unknown dtype {dtype}.")


@lru_cache(maxsize=1024)
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


@lru_cache(maxsize=1024)
def _is_shape_compatible(
    arg_shape: Tuple[Union[int, None, str], ...],
    gt_shape: Tuple[Union[int, None, str], ...],
) -> bool:
    if "..." in gt_shape:
        pre_ellipsis = None
        post_ellipsis = None

        for i, dim in enumerate(gt_shape):
            if dim == "...":
                pre_ellipsis = i
                post_ellipsis = len(gt_shape) - i - 1
                break

        if pre_ellipsis is None or gt_shape.count("...") > 1:
            raise ValueError(
                "Only one ellipsis is supported in the shape hint for now."
            )

        if len(arg_shape) < len(gt_shape) - 1:
            return False

        for i in range(pre_ellipsis):
            if arg_shape[i] != gt_shape[i] and gt_shape[i] is not None:
                return False

        for i in range(1, post_ellipsis + 1):
            if arg_shape[-i] != gt_shape[-i] and gt_shape[-i] is not None:
                return False

        return True
    else:
        if len(arg_shape) != len(gt_shape):
            return False

        for arg_dim, gt_dim in zip(arg_shape, gt_shape):
            if arg_dim != gt_dim and gt_dim is not None:
                return False

        return True


@lru_cache(maxsize=1024)
def _assert_tensor_hint(
    hint: jaxtyping.AbstractArray,
    arg_shape: Tuple[int, ...],
    arg_dtype: Any,
    arg_name: str,
):
    """
    Args:
        hint: A type hint for a tensor, must be javtyping.AbstractArray.
        arg: An argument to check, typically a tensor.
        arg_name: The name of the argument, for error messages.
    """
    # Check shapes.
    gt_shape = _shape_from_dim_str(hint.dim_str)
    if not _is_shape_compatible(arg_shape, gt_shape):
        raise TypeError(
            f"{arg_name} must be of shape {gt_shape}, but got shape {arg_shape}."
        )

    # Check dtype.
    gt_dtypes = hint.dtypes
    if _dtype_to_str(arg_dtype) not in gt_dtypes:
        raise TypeError(
            f"{arg_name} must be of dtype {gt_dtypes}, "
            f"but got dtype {_dtype_to_str(arg_dtype)}."
        )


@lru_cache(maxsize=1)
def _get_valid_array_types():
    if is_torch_available():
        valid_array_types = (np.ndarray, torch.Tensor)
    else:
        valid_array_types = (np.ndarray,)
    return valid_array_types


# Global flag to enable or disable tensor type check
_tensor_check_enabled = True


def enable_tensor_check(enabled: bool):
    """
    Enable or disable the tensor type check globally. This is useful for
    debugging purposes to disable the type check without removing the
    decorator. By default, the tensor type check is enabled.
    """
    global _tensor_check_enabled
    _tensor_check_enabled = enabled


def is_tensor_check_enabled() -> bool:
    """
    Returns True if tensor type check is enabled, otherwise False.
    """
    return _tensor_check_enabled


class Tensor:
    """
    An abstract tensor type for type hinting only.
    Typically np.ndarray or torch.Tensor is supported.
    """

    pass


def tensor_to_auto_backend(func, force_backend=None):
    """
    Automatic backend selection based on the backend of type-annotated input
    tensors, and run tensor type and shape checks if is_tensor_check_enabled().
    If there are no tensors, or if the tensors do not have the necessary type
    annotations, the default backend is used. The function targets specifically
    jaxtyping.AbstractArray annotations to determine tensor treatment and
    backend usage.

    Detailed behaviors:
    1. Only processes input arguments that are explicitly typed as
       jaxtyping.AbstractArray. Arguments without this type hint or with
       different annotations maintain their default behavior without backend
       modification.
    2. Supports handling of numpy.ndarray, torch.Tensor, and Python lists that
       should be converted to tensors based on their type hints.
    3. If the type hint is jaxtyping.AbstractArray and the argument is a list,
       the list will be converted to a tensor using the native array
       functionality of the active backend.
    4. Ensures all tensor arguments must be from the same backend to avoid
       conflicts.
    5. Uses the default backend if no tensors are present or if the tensors do
       not require specific backend handling based on their annotations.
    6. If force_backend is specified, the inferred backend from arguments
       and type hints will be ignored, and the specified backend will be used
       instead. Don't confuse this with the default backend as this takes
       higher precedence.
    """

    def _collect_tensors(args: List[Any]) -> List[Any]:
        """
        Recursively collects np.ndarray and torch.Tensor objects. Other types
        including lists are ignored. Processing lists can be slow, as we need
        to check each element for tensors.
        """
        if is_torch_available():
            tensor_types = (np.ndarray, torch.Tensor)
        else:
            tensor_types = (np.ndarray,)

        tensors = []
        for arg in args:
            if isinstance(arg, tensor_types):
                tensors.append(arg)

        return tensors

    def _determine_backend(
        arg_name_to_arg: Dict[str, Any],
        arg_name_to_hint: Dict[str, Any],
    ) -> str:
        """
        Also throws an error if the tensors are not from the same backend.
        """
        tensor_annotated_args = []
        for arg_name, hint in arg_name_to_hint.items():
            if (
                arg_name in arg_name_to_arg
                and inspect.isclass(hint)
                and issubclass(hint, jaxtyping.AbstractArray)
            ):
                arg = arg_name_to_arg[arg_name]
                tensor_annotated_args.append(arg)
        tensors = _collect_tensors(tensor_annotated_args)

        if not tensors:
            return "numpy"
        elif all(isinstance(t, np.ndarray) for t in tensors):
            return "numpy"
        elif is_torch_available() and all(isinstance(t, torch.Tensor) for t in tensors):
            return "torch"
        else:
            raise TypeError("All tensors must be from the same backend.")

    def _convert_tensor_to_backend(arg, backend):
        """
        Convert the tensor to the specified backend. It shall already be checked
        that the arg is a tensor-like object, the tensor is type-annotated, and
        the backend is valid.
        """
        if backend == "numpy":
            if isinstance(arg, np.ndarray):
                return arg
            elif is_torch_available() and isinstance(arg, torch.Tensor):
                return arg.detach().cpu().numpy()
            elif isinstance(arg, list):
                return np.array(arg)
            else:
                raise ValueError(
                    f"Unsupported type {type(arg)} for conversion to numpy."
                )
        elif backend == "torch":
            if not is_torch_available:
                raise ValueError("Torch is not available.")
            elif isinstance(arg, torch.Tensor):
                return arg
            elif isinstance(arg, np.ndarray):
                return torch.from_numpy(arg)
            elif isinstance(arg, list):
                return torch.tensor(arg)
            else:
                raise ValueError(
                    f"Unsupported type {type(arg)} for conversion to torch."
                )
        else:
            raise ValueError(f"Unsupported backend {backend}.")

    # Pre-compute the function signature and type hints
    # This is called per function declaration and not per function call
    sig = inspect.signature(func)
    hint_dict = typing.get_type_hints(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Construct the full argument dictionary from args and kwargs
        arg_names = [param.name for param in sig.parameters.values()]
        arg_dict = dict(zip(arg_names, args))
        arg_dict.update(kwargs)

        # Fill in missing arguments with their default values
        for arg_name, param in sig.parameters.items():
            if arg_name not in arg_dict and param.default is not param.empty:
                arg_dict[arg_name] = param.default

        # Determine backend
        if force_backend is None:
            backend = _determine_backend(arg_dict, hint_dict)
        elif force_backend in ("numpy", "torch"):
            backend = force_backend
        else:
            raise ValueError(f"Unsupported forced backend {force_backend}.")

        # Convert tensors to the appropriate backend
        for arg_name, hint in hint_dict.items():
            if (
                arg_name in arg_dict
                and inspect.isclass(hint)
                and issubclass(hint, jaxtyping.AbstractArray)
            ):
                arg_dict[arg_name] = _convert_tensor_to_backend(
                    arg_dict[arg_name], backend
                )

        # Check tensor dtype and shape if enabled
        if is_tensor_check_enabled():
            for arg_name, arg in arg_dict.items():
                if (
                    arg_name in hint_dict
                    and inspect.isclass(hint_dict[arg_name])
                    and issubclass(hint_dict[arg_name], jaxtyping.AbstractArray)
                ):
                    hint = hint_dict[arg_name]
                    if isinstance(arg, _get_valid_array_types()):
                        _assert_tensor_hint(
                            hint=hint,
                            arg_shape=arg.shape,
                            arg_dtype=arg.dtype,
                            arg_name=arg_name,
                        )

        # Call the original function with updated arguments
        result = func(**arg_dict)

        return result

    return wrapper


def tensor_to_numpy_backend(func):
    """
    Run this function by first converting its input tensors to numpy arrays.
    Only jaxtyping-annotated tensors will be processed. This wrapper shall be
    used if the internal implementation is numpy-only or if we expect to return
    numpy arrays.

    Behavior:
    1. Only converts arguments that are annotated explicitly with a jaxtyping
       tensor type. If the type hint is a container of tensors, the conversion
       will not be performed.
    2. Supports conversion of lists into numpy arrays if they are intended to be
       tensors, according to the function's type annotations.
    3. The conversion is applied to top-level arguments and does not recursively
       convert tensors within nested custom types (e.g., custom classes
       containing tensors).
    4. This decorator is particularly useful for functions requiring consistent
       tensor handling specifically with numpy, ensuring compatibility and
       simplifying operations that depend on numpy's functionality.

    Note:
    - The decorator inspects type annotations and applies conversions where
      specified.
    - Lists of tensors or tensors within lists annotated as tensors
      will be converted to numpy arrays if not already in that format.

    This function simply wraps the tensor_auto_backend function with the
    force_backend argument set to "numpy".
    """
    # Wrap the original function with tensor_auto_backend enforcing numpy.
    wrapped_func = tensor_to_auto_backend(func, force_backend="numpy")

    @wraps(func)
    def wrapper(*args, **kwargs):
        return wrapped_func(*args, **kwargs)

    return wrapper


def tensor_to_torch_backend(func):
    """
    Run this function by first converting its input tensors to torch tensors.
    Only jaxtyping-annotated tensors will be processed. This wrapper shall be
    used if the internal implementation is torch-only or if we expect to return
    torch tensors.

    Behavior:
    1. Only converts arguments that are annotated explicitly with a jaxtyping
       tensor type. If the type hint is a container of tensors, the conversion
       will not be performed.
    2. Supports conversion of lists into torch tensors if they are intended to be
       tensors, according to the function's type annotations.
    3. The conversion is applied to top-level arguments and does not recursively
       convert tensors within nested custom types (e.g., custom classes
       containing tensors).
    4. This decorator is particularly useful for functions requiring consistent
       tensor handling specifically with torch, ensuring compatibility and
       simplifying operations that depend on torch's functionality.

    Note:
    - The decorator inspects type annotations and applies conversions where
      specified.
    - Lists of tensors or tensors within lists annotated as tensors
      will be converted to torch tensors if not already in that format.

    This function simply wraps the tensor_auto_backend function with the
    force_backend argument set to "torch".
    """
    # Wrap the original function with tensor_auto_backend enforcing torch.
    wrapped_func = tensor_to_auto_backend(func, force_backend="torch")

    @wraps(func)
    def wrapper(*args, **kwargs):
        return wrapped_func(*args, **kwargs)

    return wrapper
