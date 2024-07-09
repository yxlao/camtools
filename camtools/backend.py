import inspect
import typing
import warnings
from functools import lru_cache, wraps
from typing import Any, Tuple, Union

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


# Global variable to keep track of the tensor type check status
_tensor_check_enabled = True


def enable_tensor_check():
    """
    Enable the tensor type check globally. This function activates type checking
    for tensors, which is useful for ensuring that tensor operations are
    performed correctly, especially during debugging and development.
    """
    global _tensor_check_enabled
    _tensor_check_enabled = True


def disable_tensor_check():
    """
    Disable the tensor type check globally. This function deactivates type checking
    for tensors, which can be useful for performance optimizations or when
    type checks are known to be unnecessary or problematic.
    """
    global _tensor_check_enabled
    _tensor_check_enabled = False


def is_tensor_check_enabled():
    """
    Returns True if the tensor dtype and shape check is enabled, and False
    otherwise. This will be used when @tensor_to_auto_backend,
    @tensor_to_numpy_backend, or @tensor_to_torch_backend is called.
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

    # Pre-compute the function signature and type hints
    # This is called per function declaration and not per function call
    sig = inspect.signature(func)
    arg_names = [
        param.name
        for param in sig.parameters.values()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    arg_name_to_hint = {
        name: hint
        for name, hint in typing.get_type_hints(func).items()
        if name in arg_names
    }
    tensor_names = [
        name
        for name, hint in arg_name_to_hint.items()
        if inspect.isclass(hint) and issubclass(hint, jaxtyping.AbstractArray)
    ]

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
            elif isinstance(arg, (list, tuple)):
                return np.array(arg)
            else:
                raise ValueError(
                    f"Unsupported type {type(arg)} for conversion to numpy."
                )
        elif backend == "torch":
            if not is_torch_available():
                raise ValueError("Torch is not available.")
            elif isinstance(arg, torch.Tensor):
                return arg
            elif isinstance(arg, np.ndarray):
                return torch.from_numpy(arg)
            elif isinstance(arg, (list, tuple)):
                return torch.tensor(arg)
            else:
                raise ValueError(
                    f"Unsupported type {type(arg)} for conversion to torch."
                )
        else:
            raise ValueError(f"Unsupported backend {backend}.")

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind args and kwargs
        # This is faster than sig.bind() but less flexible
        arg_name_to_arg = dict(zip(arg_names, args))
        arg_name_to_arg.update(kwargs)

        # Fill in missing arguments with their default values
        for arg_name, param in sig.parameters.items():
            if arg_name not in arg_name_to_arg and param.default is not param.empty:
                arg_name_to_arg[arg_name] = param.default

        # Determine backend
        if force_backend is None:
            # Recursively collect np.ndarray and torch.Tensor objects
            # Other types including lists are ignored
            if is_torch_available():
                tensor_types = (np.ndarray, torch.Tensor)
            else:
                tensor_types = (np.ndarray,)
            tensors = [
                arg_name_to_arg[tensor_name]
                for tensor_name in tensor_names
                if isinstance(arg_name_to_arg[tensor_name], tensor_types)
            ]

            # Determine the backend based on tensor types present
            if not tensors:
                backend = "numpy"
            else:
                tensor_types_used = {type(t) for t in tensors}
                if tensor_types_used == {np.ndarray}:
                    backend = "numpy"
                elif is_torch_available() and tensor_types_used == {torch.Tensor}:
                    backend = "torch"
                else:
                    raise TypeError(
                        f"All tensors must be from the same backend, "
                        f"but got {tensor_types_used}."
                    )
        elif force_backend in ("numpy", "torch"):
            backend = force_backend
        else:
            raise ValueError(f"Unsupported forced backend {force_backend}.")

        # Convert tensors to the appropriate backend
        for tensor_name in tensor_names:
            arg_name_to_arg[tensor_name] = _convert_tensor_to_backend(
                arg_name_to_arg[tensor_name], backend
            )

        # Check tensor dtype and shape if enabled
        if is_tensor_check_enabled():
            for tensor_name in tensor_names:
                hint = arg_name_to_hint[tensor_name]
                tensor_arg = arg_name_to_arg[tensor_name]
                if isinstance(tensor_arg, _get_valid_array_types()):
                    _assert_tensor_hint(
                        hint=hint,
                        arg_shape=tensor_arg.shape,
                        arg_dtype=tensor_arg.dtype,
                        arg_name=tensor_name,
                    )

        # Call the original function with updated arguments
        result = func(**arg_name_to_arg)

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
