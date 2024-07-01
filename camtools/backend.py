import typing
import warnings
from functools import lru_cache, wraps
from inspect import signature
from typing import Literal

import ivy
import jaxtyping

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


def with_native_backend(func):
    """
    1. Enable default camtools backend.
    2. Return native backend array (setting array mode to False).
    3. Converts lists to tensors if the type hint is a tensor.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        og_backend = ivy.current_backend()
        ct_backend = get_backend()
        ivy.set_backend(ct_backend)

        # Unpack args and type hints
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        arg_name_to_hint = typing.get_type_hints(func)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                with ivy.ArrayMode(False):
                    # Convert list -> native tensor if the type hint is a tensor
                    for arg_name, arg in bound_args.arguments.items():
                        if arg_name in arg_name_to_hint and issubclass(
                            arg_name_to_hint[arg_name], jaxtyping.AbstractArray
                        ):
                            if isinstance(arg, list):
                                bound_args.arguments[arg_name] = ivy.native_array(arg)

                    # Call the function
                    result = func(*bound_args.args, **bound_args.kwargs)
        finally:
            ivy.set_backend(og_backend)
        return result

    return wrapper
