import numpy as np
import pytest
import camtools as ct
import ivy
from jaxtyping import Float
from camtools.typing import Tensor


def is_torch_available():
    try:
        import torch

        return True
    except ImportError:
        return False


@ct.backend.with_auto_backend
def concat_tensors(x: Float[Tensor, "..."], y: Float[Tensor, "..."]):
    return ivy.concat([x, y], axis=0)


def test_no_tensor_default_backend_numpy():
    """
    Test the default backend when no tensors are provided.
    """
    result = concat_tensors([1, 2, 3], [4, 5, 6])
    assert isinstance(result, np.ndarray), "Expected numpy array with no tensor inputs."
    assert np.array_equal(
        result, np.array([1, 2, 3, 4, 5, 6])
    ), "Array contents mismatch."


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_no_tensor_default_backend_torch():
    """
    Test the default backend when no tensors are provided.
    """
    import torch

    with ct.backend.ScopedBackend("torch"):
        result = concat_tensors([1, 2, 3], [4, 5, 6])
        assert isinstance(
            result, torch.Tensor
        ), "Expected torch tensor with no tensor inputs."
    assert torch.equal(
        result, torch.tensor([1, 2, 3, 4, 5, 6])
    ), "Tensor contents mismatch."


def test_pure_list_as_tensor():
    """
    Test handling of pure Python lists annotated as tensor type.
    """
    ct.backend.set_backend("numpy")

    @ct.backend.with_auto_backend
    def func(x: Float[Tensor, "..."]):
        return ivy.array(x)  # Convert list to tensor based on backend

    result = func([1.0, 2.0, 3.0])
    assert isinstance(
        result, np.ndarray
    ), "Expected numpy array when lists are type-annotated as tensors."


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_mixed_tensor_types():
    """
    Test error handling with mixed tensor types across arguments.
    """
    import torch

    numpy_data = np.array([1, 2, 3])
    torch_data = torch.tensor([4, 5, 6])
    with pytest.raises(TypeError):
        concat_tensors(numpy_data, torch_data)


def test_container_of_tensors():
    """
    Test handling of containers holding tensors from different backends.
    """
    if is_torch_available():
        import torch

        numpy_data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        torch_data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        with pytest.raises(TypeError):
            # This should fail as tensors from different backends are mixed in containers
            concat_tensors(numpy_data, torch_data)


def test_ivy_array_mode():
    """
    Ensure ivy.ArrayMode(False) is applied within the function.
    """
    ct.backend.set_backend("numpy")
    with ivy.ArrayMode() as mode:
        concat_tensors([1, 2, 3])
        assert (
            not mode.is_array_mode
        ), "ivy.ArrayMode should be set to False within the function."


@pytest.fixture(autouse=True)
def restore_backend():
    """
    Fixture to reset the backend after each test.
    """
    yield
    ct.backend.set_backend(
        "numpy"
    )  # Reset to numpy after each test to avoid state leakage


# Additional tests to cover diverse containers and type annotation scenarios
@pytest.mark.parametrize("container", [list, tuple, set])
def test_tensor_containers(container):
    """
    Test with different collections of tensors.
    """
    ct.backend.set_backend("numpy")
    numpy_tensor = np.ones((3, 3))
    if container in {
        set,
    }:
        # Sets do not support indexing
        # They can't hold multiple mutable identical elements
        with pytest.raises(TypeError):
            concat_tensors(container([numpy_tensor, numpy_tensor]))
    else:
        result = concat_tensors(container([numpy_tensor, numpy_tensor * 2]))
        assert result.shape == (
            2,
            3,
            3,
        ), "Shape mismatch for tensor container processing."


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_invalid_tensor_collections():
    """
    Test invalid collections with mixed backend tensors.
    """
    import torch

    numpy_tensor = np.ones((3, 3))
    torch_tensor = torch.ones((3, 3))
    with pytest.raises(TypeError):
        concat_tensors([numpy_tensor, torch_tensor])  # Mixed backends in a list
    with pytest.raises(TypeError):
        concat_tensors((numpy_tensor, torch_tensor))  # Mixed backends in a tuple
