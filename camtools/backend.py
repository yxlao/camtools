from typing import Literal
from functools import wraps
import ivy

_default_backend = "numpy"


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
    Decorator to run a function with: 1) default camtools backend, 2) with
    native backend array (setting array mode to False).
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        og_backend = ivy.current_backend()
        ct_backend = get_backend()
        ivy.set_backend(ct_backend)
        try:
            with ivy.ArrayMode(False):
                result = func(*args, **kwargs)
        finally:
            ivy.set_backend(og_backend)
        return result

    return wrapper
