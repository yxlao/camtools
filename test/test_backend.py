"""
Test basic usage of ivy and its interaction with numpy and torch.
"""

import ivy
import numpy as np
import torch
import einops
import camtools as ct
from jaxtyping import Array, Float, UInt8


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
    add(src_x, src_y)


def test_type_hint_arguments():
    """
    Test type hinting arguments.
    """

    @ct.backend.with_native_backend
    def add(x: Float[Array, 2, 3], y: Float[Array, 1, 3]) -> Float[Array, 2, 3]:
        return x + y
