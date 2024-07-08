import numpy as np
import pytest
from jaxtyping import Float

import camtools as ct
from camtools.backend import Tensor, ivy, is_torch_available, torch
import warnings


@pytest.fixture(autouse=True)
def ignore_ivy_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*Compositional function.*array_mode is set to False.*",
        category=UserWarning,
    )
    yield


@ct.backend.tensor_to_auto_backend
def concat(x: Float[Tensor, "..."], y: Float[Tensor, "..."]):
    return ivy.concat([x, y], axis=0)


def test_concat_numpy():
    """
    Test the default backend when no tensors are provided.
    """
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    result = concat(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_torch():
    """
    Test the default backend when no tensors are provided.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = concat(x, y)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_concat_list_to_numpy():
    """
    Test the default backend when no tensors are provided.
    """
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    result = concat(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_concat_mix_list_and_numpy():
    """
    Test handling of mixed list and tensor types.
    """
    x = [1.0, 2.0, 3.0]
    y = np.array([4.0, 5.0, 6.0])
    result = concat(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_mix_list_and_torch():
    """
    Test handling of mixed list and tensor types.
    """
    x = [1.0, 2.0, 3.0]
    y = torch.tensor([4.0, 5.0, 6.0])
    result = concat(x, y)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_mix_numpy_and_torch():
    """
    Test error handling with mixed tensor types across arguments.
    """
    x = np.array([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    with pytest.raises(TypeError, match=r".*must be from the same backend.*"):
        concat(x, y)


def test_concat_list_of_numpy():
    """
    Test handling of containers holding tensors from different backends.
    """

    x = [np.array(1.0), np.array(2.0), np.array(3.0)]
    y = np.array([4.0, 5.0, 6.0])
    result = concat(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_list_of_torch():
    """
    Test handling of containers holding tensors from different backends.
    """
    x = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    y = torch.tensor([4.0, 5.0, 6.0])
    result = concat(x, y)

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_list_of_numpy_and_torch():
    """
    Test handling with mixed tensor types across containers.

    In this case as lists are not type-checked, we both x and y will be
    converted to default backend's arrays internally. That is,
    x <- np.array(x) and y <- np.array(y) are both valid operation. In this
    case, even though y contains tensors from both numpy and torch, as
    np.asarray(y) is valid, the function should work.

    However, this can be very slow. As creating a torch tensor from a list of
    np.ndarray is very slow and likewise for creating np.ndarray from a list of
    torch tensors. Therefore, you shall avoid doing this in practice.
    """
    x = [np.array(1.0), np.array(2.0), np.array(3.0)]
    y = [torch.tensor(4.0), torch.tensor(5.0), torch.tensor(6.0)]
    result = concat(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_creation():
    @ct.backend.tensor_to_auto_backend
    def creation():
        zeros = ivy.zeros([2, 3])
        return zeros

    # Default backend is numpy
    tensor = creation()
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (2, 3)
    assert tensor.dtype == np.float32


def test_type_hint_arguments_numpy():
    @ct.backend.tensor_to_auto_backend
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
    @ct.backend.tensor_to_auto_backend
    def add(
        x: Float[Tensor, "2 3"],
        y: Float[Tensor, "1 3"],
    ) -> Float[Tensor, "2 3"]:
        return x + y

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


def test_named_dim_numpy():
    @ct.backend.tensor_to_auto_backend
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
    @ct.backend.tensor_to_auto_backend
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


def test_concat_tensors_with_numpy():
    @ct.backend.tensor_to_numpy_backend
    def concat_tensors_with_numpy(
        x: Float[Tensor, "..."],
        y: Float[Tensor, "..."],
    ):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.concatenate([x, y], axis=0)

    # Test with numpy arrays
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    result = concat_tensors_with_numpy(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Test with torch tensors
    if is_torch_available():
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        result = concat_tensors_with_numpy(x, y)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Test with lists
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    result = concat_tensors_with_numpy(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Numpy and list mixed
    x = np.array([1.0, 2.0, 3.0])
    y = [4.0, 5.0, 6.0]
    result = concat_tensors_with_numpy(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Torch and list mixed
    if is_torch_available():
        x = torch.tensor([1.0, 2.0, 3.0])
        y = [4.0, 5.0, 6.0]
        result = concat_tensors_with_numpy(x, y)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_tensors_with_torch():
    @ct.backend.tensor_to_torch_backend
    def concat_tensors_with_torch(
        x: Float[Tensor, "..."],
        y: Float[Tensor, "..."],
    ):
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        return torch.cat([x, y], axis=0)

    # Test with numpy arrays
    x_np = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    y_np = np.array([4.0, 5.0, 6.0]).astype(np.float32)
    result_np = concat_tensors_with_torch(x_np, y_np)
    assert isinstance(result_np, torch.Tensor)
    assert torch.allclose(result_np, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Test with torch tensors
    x_torch = torch.tensor([1.0, 2.0, 3.0])
    y_torch = torch.tensor([4.0, 5.0, 6.0])
    result_torch = concat_tensors_with_torch(x_torch, y_torch)
    assert isinstance(result_torch, torch.Tensor)
    assert torch.equal(result_torch, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Test with lists
    x_list = [1.0, 2.0, 3.0]
    y_list = [4.0, 5.0, 6.0]
    result_list = concat_tensors_with_torch(x_list, y_list)
    assert isinstance(result_list, torch.Tensor)
    assert torch.allclose(result_list, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Mixed types: numpy array and list
    x_mixed = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    y_mixed = [4.0, 5.0, 6.0]
    result_mixed = concat_tensors_with_torch(x_mixed, y_mixed)
    assert isinstance(result_mixed, torch.Tensor)
    assert torch.allclose(result_mixed, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    # Mixed types: torch tensor and list
    x_mixed_torch = torch.tensor([1.0, 2.0, 3.0])
    y_mixed_list = [4.0, 5.0, 6.0]
    result_mixed_torch_list = concat_tensors_with_torch(x_mixed_torch, y_mixed_list)
    assert isinstance(result_mixed_torch_list, torch.Tensor)
    assert torch.allclose(
        result_mixed_torch_list, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )
