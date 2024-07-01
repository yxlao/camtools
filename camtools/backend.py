import collections.abc
import typing
import warnings
from functools import lru_cache, wraps
from inspect import signature
from typing import Any, Literal

import ivy
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


def with_auto_backend(func):
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
    """

    def _collect_tensors(item: Any, tensors: list) -> None:
        """
        Recursively collects tensors from nested iterable structures.
        """
        if isinstance(item, np.ndarray):
            tensors.append(item)
        elif is_torch_available():
            import torch

            if isinstance(item, torch.Tensor):
                tensors.append(item)
        elif isinstance(item, collections.abc.Iterable) and not isinstance(
            item, (str, bytes)
        ):
            for subitem in item:
                _collect_tensors(subitem, tensors)

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
        raise TypeError("All tensors must be from the same backend (numpy or torch).")

    @wraps(func)
    def wrapper(*args, **kwargs):
        stashed_backend = ivy.current_backend()
        tensors = []

        # Unpack args and kwargs and collect all tensors
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for arg in bound_args.arguments.values():
            _collect_tensors(arg, tensors)

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
