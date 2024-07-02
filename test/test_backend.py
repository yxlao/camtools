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


def test_default_backend_numpy():
    """
    Test the default backend when no tensors are provided.
    """
    result = concat_tensors([1, 2, 3], [4, 5, 6])
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_default_backend_torch():
    """
    Test the default backend when no tensors are provided.
    """
    import torch

    with ct.backend.ScopedBackend("torch"):
        result = concat_tensors([1, 2, 3], [4, 5, 6])
        assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1, 2, 3, 4, 5, 6]))


def test_pure_list_as_tensor_numpy():
    """
    Test handling of pure Python lists annotated as tensor type.
    """

    @ct.backend.with_auto_backend
    def func(x: Float[Tensor, "..."]):
        return ivy.native_array(x)

    result = func([1.0, 2.0, 3.0])
    assert isinstance(result, np.ndarray)


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_pure_list_as_tensor_torch():
    """
    Test handling of pure Python lists annotated as tensor type.
    """
    import torch

    @ct.backend.with_auto_backend
    def func(x: Float[Tensor, "..."]):
        return ivy.native_array(x)

    with ct.backend.ScopedBackend("torch"):
        result = func([1.0, 2.0, 3.0])
        assert isinstance(result, torch.Tensor)


def test_mix_list_and_tensor_types_numpy():
    """
    Test handling of mixed list and tensor types.
    """
    numpy_data = np.array([1, 2, 3])
    list_data = [4, 5, 6]
    result = concat_tensors(numpy_data, list_data)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6]))


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

        # Tensors from different backends are mixed in containers
        with pytest.raises(TypeError):
            concat_tensors(numpy_data, torch_data)


def test_ivy_array_mode():
    """
    Ensure ivy.ArrayMode(False) is applied within the function.
    """
    with ivy.ArrayMode() as mode:
        concat_tensors([1, 2, 3])
        assert not mode.is_array_mode


# Additional tests to cover diverse containers and type annotation scenarios
@pytest.mark.parametrize("container", [list, tuple, set])
def test_tensor_containers(container):
    """
    Test with different collections of tensors.
    """
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
        assert result.shape == (2, 3, 3)


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_invalid_tensor_collections():
    """
    Test invalid collections with mixed backend tensors.
    """
    import torch

    numpy_tensor = np.ones((3, 3))
    torch_tensor = torch.ones((3, 3))

    # Mixed backends in a list
    with pytest.raises(TypeError):
        concat_tensors([numpy_tensor, torch_tensor])

    # Mixed backends in a tuple
    with pytest.raises(TypeError):
        concat_tensors((numpy_tensor, torch_tensor))