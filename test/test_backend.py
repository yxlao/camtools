"""
Test basic usage of ivy and its interaction with numpy and torch.
"""

import typing
from functools import wraps
from typing import Union

import numpy as np
import pytest
import torch
from jaxtyping import Float, UInt8, _array_types

import camtools as ct
from camtools.backend import ivy
from camtools.typing import Tensor


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


def test_type_hint_arguments():
    """
    Test type hinting arguments.
    """

    @ct.backend.with_native_backend
    @ct.typing.check_shape_and_dtype
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

    # List can be converted to numpy automatically
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = add(x, y)
    assert np.allclose(result, expected, atol=1e-5)

    # Incorrect shapes
    with pytest.raises(TypeError):
        y_wrong = np.array([[1, 1, 1, 1]], dtype=np.float32)
        add(x, y_wrong)

    # Incorrect shape with lists
    with pytest.raises(TypeError):
        y_wrong = [[1.0, 1.0, 1.0, 1.0]]
        add(x, y_wrong)

    # Incorrect dtype
    with pytest.raises(TypeError):
        y_wrong = np.array([[1, 1, 1]], dtype=np.int64)
        add(x, y_wrong)

    # Incorrect dtype with lists
    with pytest.raises(TypeError):
        y_wrong = [[1, 1, 1]]
        add(x, y_wrong)
