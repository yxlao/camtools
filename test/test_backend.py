"""
Test basic usage of ivy and its interaction with numpy and torch.
"""

import ivy
import numpy as np
import torch
import einops
import camtools as ct
from jaxtyping import Float, UInt8
import typing
import pytest
from numpy.typing import NDArray

from functools import wraps
from jaxtyping import Float, _array_types

from typing import Union


def test_creation():
    """
    Test tensor creation.
    """

    @ct.backend.with_native_backend
    def creation():
        zeros = ivy.zeros([2, 3])
        return zeros

    # Default backend is numpy
    assert ct.backend.get_backend() == "numpy"
    tensor = creation()
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (2, 3)
    assert tensor.dtype == np.float32

    # Switch to torch backend
    ct.backend.set_backend("torch")
    assert ct.backend.get_backend() == "torch"
    tensor = creation()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 3)
    assert tensor.dtype == torch.float32
    ct.backend.set_backend("numpy")


def test_arguments():
    """
    Test taking arguments from functions.
    """

    @ct.backend.with_native_backend
    def add(x, y):
        return x + y

    # Default backend is numpy
    assert ct.backend.get_backend() == "numpy"
    src_x = np.ones([2, 3]) * 2
    src_y = np.ones([1, 3]) * 3
    dst_expected = np.ones([2, 3]) * 5
    dst = add(src_x, src_y)
    np.testing.assert_allclose(dst, dst_expected, rtol=1e-5, atol=1e-5)

    # Mixed backend argument should raise error
    src_x = np.ones([2, 3]) * 2
    src_y = torch.ones([1, 3]) * 3
    with pytest.raises(TypeError):
        add(src_x, src_y)


def check_shape_and_dtype(func):
    """
    A decorator to enforce type and shape specifications as per type hints.
    """

    def get_shape(dims):
        shape = []
        for dim in dims:
            if isinstance(dim, _array_types._FixedDim):
                shape.append(dim.size)
            elif isinstance(dim, _array_types._NamedDim):
                shape.append(None)
        return tuple(shape)

    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = typing.get_type_hints(func)
        arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]

        for arg_name, arg_value in zip(arg_names, args):
            if arg_name in hints:
                hint = hints[arg_name]
                expected_shape = get_shape(hint.dims)

                if not (isinstance(arg_value, (np.ndarray, torch.Tensor))):
                    raise TypeError(f"{arg_name} must be a tensor")

                if not all(
                    actual_dim == expected_dim or expected_dim is None
                    for (
                        actual_dim,
                        expected_dim,
                    ) in zip(
                        arg_value.shape,
                        expected_shape,
                    )
                ):
                    raise TypeError(
                        f"{arg_name} must be a tensor of shape {expected_shape}"
                    )

        return func(*args, **kwargs)

    return wrapper


Tensor = Union[np.ndarray, torch.Tensor]


def test_type_hint_arguments():
    """
    Test type hinting arguments.
    """

    @ct.backend.with_native_backend
    @ct.sanity.check_shape_and_dtype
    def add(
        x: Float[Tensor, "2 3"],
        y: Float[Tensor, "1 3"],
    ) -> Float[Tensor, "2 3"]:
        return x + y

    # Default backend is numpy
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([[1, 1, 1]], dtype=np.float32)
    result = add(x, y)
    expected = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
    assert np.allclose(result, expected, atol=1e-5)

    # Testing with incorrect shapes
    with pytest.raises(TypeError):
        y_wrong = np.array([[1, 1, 1, 1]], dtype=np.float32)
        add(x, y_wrong)

    # Testing with incorrect types
    with pytest.raises(TypeError):
        x_wrong_type = [[1, 2, 3], [4, 5, 6]]  # not a NumPy array
        add(x_wrong_type, y)
