import numpy as np
import pytest
from jaxtyping import Float

import camtools as ct
from camtools.backend import ivy, is_torch_available
from camtools.typing import Tensor


def test_creation_numpy():
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


@pytest.mark.skipif(not ct.backend.is_torch_available(), reason="Skip torch")
def test_creation_torch():
    import torch

    @ct.backend.with_native_backend
    def creation():
        zeros = ivy.zeros([2, 3])
        return zeros

    # Switch to torch backend
    ct.backend.set_backend("torch")
    assert ct.backend.get_backend() == "torch"
    tensor = creation()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 3)
    assert tensor.dtype == torch.float32
    ct.backend.set_backend("numpy")


def test_arguments_numpy():
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


@pytest.mark.skipif(not ct.backend.is_torch_available(), reason="Skip torch")
def test_arguments_torch():
    import torch

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


def test_type_hint_arguments_numpy():
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
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, atol=1e-5)

    # List can be converted to numpy automatically
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = add(x, y)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, atol=1e-5)

    # Incorrect shapes
    with pytest.raises(TypeError, match=r".*but got shape.*"):
        y_wrong = np.array([[1, 1, 1, 1]], dtype=np.float32)
        add(x, y_wrong)

    # Incorrect shape with lists
    with pytest.raises(TypeError, match=r".*but got shape.*"):
        y_wrong = [[1.0, 1.0, 1.0, 1.0]]
        add(x, y_wrong)

    # Incorrect dtype
    with pytest.raises(TypeError, match=r".*but got dtype.*"):
        y_wrong = np.array([[1, 1, 1]], dtype=np.int64)
        add(x, y_wrong)

    # Incorrect dtype with lists
    with pytest.raises(TypeError, match=r".*but got dtype.*"):
        y_wrong = [[1, 1, 1]]
        add(x, y_wrong)


@pytest.mark.skipif(not ct.backend.is_torch_available(), reason="Skip torch")
def test_type_hint_arguments_torch():
    import torch

    @ct.backend.with_native_backend
    @ct.typing.check_shape_and_dtype
    def add(
        x: Float[Tensor, "2 3"],
        y: Float[Tensor, "1 3"],
    ) -> Float[Tensor, "2 3"]:
        return x + y

    # With torch backend
    ct.backend.set_backend("torch")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    y = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    result = add(x, y)
    expected = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float32)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-5)

    # List can be converted to torch automatically
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = add(x, y)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-5)

    # Incorrect shapes
    with pytest.raises(TypeError, match=r".*but got shape.*"):
        y_wrong = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)
        add(x, y_wrong)

    # Incorrect shape with lists
    with pytest.raises(TypeError, match=r".*but got shape.*"):
        y_wrong = [[1.0, 1.0, 1.0, 1.0]]
        add(x, y_wrong)

    # Incorrect dtype
    with pytest.raises(TypeError, match=r".*but got dtype.*"):
        y_wrong = torch.tensor([[1, 1, 1]], dtype=torch.int64)
        add(x, y_wrong)

    # Incorrect dtype with lists
    with pytest.raises(TypeError, match=r".*but got dtype.*"):
        y_wrong = [[1, 1, 1]]
        add(x, y_wrong)

    ct.backend.set_backend("numpy")


def test_named_dim_numpy():
    @ct.backend.with_native_backend
    @ct.typing.check_shape_and_dtype
    def add(
        x: Float[Tensor, "3"],
        y: Float[Tensor, "n 3"],
    ) -> Float[Tensor, "n 3"]:
        return x + y

    # Fixed x tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Valid y tensor with shape (1, 3)
    y = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    result = add(x, y)
    expected = np.array([[5.0, 7.0, 9.0]], dtype=np.float32)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, atol=1e-5)

    # Valid y tensor with shape (2, 3)
    y = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    result = add(x, y)
    expected = np.array([[5.0, 7.0, 9.0], [8.0, 10.0, 12.0]], dtype=np.float32)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, atol=1e-5)

    # Test for a shape mismatch where y does not conform to "n 3"
    with pytest.raises(TypeError, match=r".*but got shape.*"):
        y_wrong = np.array([4.0, 5.0, 6.0], dtype=np.float32)  # Shape (3,)
        add(x, y_wrong)

    # List inputs that should be automatically converted and work
    y = [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    result = add(x, y)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected, atol=1e-5)

    # Incorrect dtype with lists, expect dtype error
    with pytest.raises(TypeError, match=r".*but got dtype.*"):
        y_wrong = [[4, 5, 6], [7, 8, 9]]  # int type elements in list
        add(x, y_wrong)


@pytest.mark.skipif(not ct.backend.is_torch_available(), reason="Skip torch")
def test_named_dim_torch():
    import torch

    @ct.backend.with_native_backend
    @ct.typing.check_shape_and_dtype
    def add(
        x: Float[Tensor, "3"],
        y: Float[Tensor, "n 3"],
    ) -> Float[Tensor, "n 3"]:
        return x + y

    # Fixed x tensor for Torch
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    # Valid y tensor with shape (1, 3)
    y = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    result = add(x, y)
    expected = torch.tensor([[5.0, 7.0, 9.0]], dtype=torch.float32)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-5)

    # Valid y tensor with shape (2, 3)
    y = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32)
    result = add(x, y)
    expected = torch.tensor([[5.0, 7.0, 9.0], [8.0, 10.0, 12.0]], dtype=torch.float32)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-5)

    # Test for a shape mismatch where y does not conform to "n 3"
    with pytest.raises(TypeError, match=r".*but got shape.*"):
        y_wrong = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)  # Shape (3,)
        add(x, y_wrong)

    # List inputs that should be automatically converted and work
    y = [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    result = add(x, y)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-5)

    # Incorrect dtype with lists, expect dtype error
    with pytest.raises(TypeError, match=r".*but got dtype.*"):
        y_wrong = [[4, 5, 6], [7, 8, 9]]  # int type elements in list
        add(x, y_wrong)
