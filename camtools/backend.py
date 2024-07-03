import collections.abc
import inspect
import typing
import warnings
from functools import lru_cache, wraps
from inspect import signature
from typing import Any, Literal, Tuple, Union

import jaxtyping
import numpy as np

_default_backend = "numpy"


@lru_cache(maxsize=None)
def _safely_import_torch_before_open3d():
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
        import open3d
    except ImportError:
        pass

    try:
        import torch

        return torch
    except ImportError:
        return None


torch = _safely_import_torch_before_open3d()


@lru_cache(maxsize=None)
def is_torch_available():
    return _safely_import_torch_before_open3d() is not None


@lru_cache(maxsize=None)
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
        category=DeprecationWarning,
        message=".*numpy.core.numeric is deprecated.*",
    )
    return __import__("ivy")


ivy = _safely_import_ivy()


class Tensor:
    """
    An abstract tensor type for type hinting only.
    Typically np.ndarray or torch.Tensor is supported.
    """

    pass


def set_backend(backend: Literal["numpy", "torch"]) -> None:
    """
    Set the default backend for camtools.
    """
    global _default_backend
    _default_backend = backend


def get_backend() -> str:
    """
    Get the default backend for camtools.
    """
    return _default_backend


class ScopedBackend:
    """
    Context manager to temporarily set the backend for camtools.

    Example:
    ```python
    with ct.backend.ScopedBackend("torch"):
        # Code that uses torch backend here
        pass

    # Code that uses the default backend here
    ```
    """

    def __init__(self, target_backend: Literal["numpy", "torch"]):
        self.target_backend = target_backend

    def __enter__(self):
        self.stashed_backend = get_backend()
        set_backend(self.target_backend)

    def __exit__(self, exc_type, exc_value, traceback):
        set_backend(self.stashed_backend)


def tensor_auto_backend(func):
    """
    Automatic backend selection for camtools functions.

    1. (Backend selection) If the function arguments does not contain any
       tensors, or only contains pure Python list and type-annotated as tensor,
       the default ct.get_backend() is used for internal computation and return
       value.
    2. (Backend selection) If the function arguments contains at least one
       tensor, the corresponding backend is used for internal computation and
       return value. The arguments can only contain tensors from one backend,
       including tensors in nested lists, otherwise an error will be raised.
    3. This wrapper will attempt to convert Python lists to tensors if the type
       hint says it should be a tensor with jaxtyping.
    4. This wrapper will set ivy.ArrayMode(False) within the function context.
    5. The automatic backend conversion is not recursive, meaning that the
       backend selection is only based on the top-level arguments. If the type
       hint is a container of tensors, the backend selection will not be applied
       to the nested tensors. For example, Float[Tensor, "..."] will be handled,
       while List[Float[Tensor, "..."]] will not be handled.
    """

    def _collect_tensors(item: Any) -> list:
        """
        Recursively collects tensors from nested iterable structures.
        The function splits logic based on whether PyTorch is available.
        """
        tensors = []
        stack = [item]

        if is_torch_available():
            import torch

            tensor_types = (np.ndarray, torch.Tensor)
        else:
            tensor_types = (np.ndarray,)

        while stack:
            current_item = stack.pop()
            if isinstance(current_item, tensor_types):
                tensors.append(current_item)
            elif isinstance(current_item, collections.abc.Iterable) and not isinstance(
                current_item, (str, bytes)
            ):
                stack.extend(current_item)

        return tensors

    def _determine_backend(tensors: list) -> str:
        """
        Determines the backend based on the types of tensors found.
        """
        if not tensors:
            return get_backend()

        if all(isinstance(t, np.ndarray) for t in tensors):
            return "numpy"
        elif is_torch_available():
            import torch

            if all(isinstance(t, torch.Tensor) for t in tensors):
                return "torch"

        raise TypeError("All tensors must be from the same backend.")

    @wraps(func)
    def wrapper(*args, **kwargs):
        stashed_backend = ivy.current_backend()

        # Unpack args and kwargs and collect all tensors
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        tensors = []
        for arg in bound_args.arguments.values():
            tensors.extend(_collect_tensors(arg))

        arg_backend = _determine_backend(tensors)
        ivy.set_backend(arg_backend)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                with ivy.ArrayMode(False):
                    # Convert list -> native tensor if the type hint is a tensor
                    for arg_name, arg in bound_args.arguments.items():
                        if (
                            arg_name in typing.get_type_hints(func)
                            and inspect.isclass(typing.get_type_hints(func)[arg_name])
                            and issubclass(
                                typing.get_type_hints(func)[arg_name],
                                jaxtyping.AbstractArray,
                            )
                            and isinstance(arg, list)
                        ):
                            bound_args.arguments[arg_name] = ivy.native_array(arg)

                    # Call the function
                    result = func(*bound_args.args, **bound_args.kwargs)
        finally:
            ivy.set_backend(stashed_backend)
        return result

    return wrapper


def tensor_type_check(func):
    """
    A decorator to enforce type and shape specifications as per type hints.

    The checks will only be performed if the tensor's type hint is exactly
    jaxtyping.AbstractArray. If it is a container of tensors, the check will
    not be performed. For example, Float[Tensor, "..."] will be checked, while
    List[Float[Tensor, "..."]] will not be checked.
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
                if inspect.isclass(hint) and issubclass(hint, jaxtyping.AbstractArray):
                    _assert_tensor_hint(hint, arg, arg_name)

        return func(*args, **kwargs)

    return wrapper


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
    if is_torch_available():
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
