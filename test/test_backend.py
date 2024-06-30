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


# def test_type_hint_arguments():
#     """
#     Test type hinting arguments.
#     """

#     @ct.backend.with_native_backend
#     def add(
#         x: Float[np.ndarray, "2 3"], y: Float[np.ndarray, "1 3"]
#     ) -> Float[np.ndarray, "2 3"]:
#         # Extract type hints
#         hints = typing.get_type_hints(add)
#         x_hint = hints["x"]
#         y_hint = hints["y"]

#         # Extract shapes from the type hints
#         x_shape = einops.parse_shape(x, x_hint)
#         y_shape = einops.parse_shape(y, y_hint)

#         # Verify the input types and shapes
#         if not (isinstance(x, (np.ndarray, torch.Tensor)) and x.shape == x_shape):
#             raise TypeError(f"x must be a tensor of shape {x_shape}")
#         if not (isinstance(y, (np.ndarray, torch.Tensor)) and y.shape == y_shape):
#             raise TypeError(f"y must be a tensor of shape {y_shape}")

#         return x + y

#     # Test with correct types and shapes using np.array directly marked as float32
#     x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
#     y = np.array([[1, 1, 1]], dtype=np.float32)
#     result = add(x, y)
#     expected = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
#     assert np.allclose(result, expected, atol=1e-5)

#     # Testing with incorrect shapes
#     with pytest.raises(TypeError):
#         x_wrong = np.array([[1, 2], [4, 5]], dtype=np.float32)
#         add(x_wrong, y)

#     # Testing with incorrect types
#     with pytest.raises(TypeError):
#         x_wrong_type = [[1, 2, 3], [4, 5, 6]]
#         add(x_wrong_type, y)
