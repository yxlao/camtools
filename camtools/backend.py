import collections.abc
import typing
import warnings
from functools import lru_cache, wraps
from inspect import signature
from typing import Any, Literal, Union
import inspect

import jaxtyping
import numpy as np

# Internally use "from camtools.backend import ivy" to make sure ivy is imported
# after the warnings filter is set. This is a temporary workaround to suppress
# the deprecation warning from numpy 2.0.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*numpy.core.numeric is deprecated.*",
)
import ivy

_default_backend = "numpy"


@lru_cache(maxsize=None)
def is_torch_available():
    try:
        import torch

        return True
    except ImportError:
        return False


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


def with_tensor_auto_backend(func):
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
                        hint = typing.get_type_hints(func)[arg_name]
                        if (
                            arg_name in typing.get_type_hints(func)
                            and inspect.isclass(hint)
                            and issubclass(
                                hint,
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
