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
@ct.typing.check_shape_and_dtype
def concat_tensors(x: Float[Tensor, "..."], y: Float[Tensor, "..."]):
    return ivy.concat([x, y], axis=0)


def test_default_backend_numpy():
    """
    Test the default backend when no tensors are provided.
    """
    result = concat_tensors([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_default_backend_torch():
    """
    Test the default backend when no tensors are provided.
    """
    import torch

    with ct.backend.ScopedBackend("torch"):
        result = concat_tensors([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


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


def test_mix_list_and_numpy():
    """
    Test handling of mixed list and tensor types.
    """
    x = np.array([1.0, 2.0, 3.0])
    y = [4.0, 5.0, 6.0]
    result = concat_tensors(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_mix_list_and_torch():
    """
    Test handling of mixed list and tensor types.
    """
    import torch

    x = torch.tensor([1.0, 2.0, 3.0])
    y = [4.0, 5.0, 6.0]

    with ct.backend.ScopedBackend("torch"):
        result = concat_tensors(x, y)

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_mix_numpy_and_torch():
    """
    Test error handling with mixed tensor types across arguments.
    """
    import torch

    x = np.array([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    with pytest.raises(TypeError, match=r".*must be from the same backend.*"):
        concat_tensors(x, y)


def test_container_of_tensors_numpy():
    """
    Test handling of containers holding tensors from different backends.
    """

    x = [np.array(1.0), np.array(2.0), np.array(3.0)]
    y = np.array([4.0, 5.0, 6.0])
    result = concat_tensors(x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_container_of_tensors_torch():
    """
    Test handling of containers holding tensors from different backends.
    """
    import torch

    x = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    y = torch.tensor([4.0, 5.0, 6.0])
    with ct.backend.ScopedBackend("torch"):
        result = concat_tensors(x, y)

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_container_mix_numpy_torch_v1():
    """
    Test error handling with mixed tensor types across containers.
    """
    import torch

    x = np.array([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    with pytest.raises(TypeError, match=r".*must be from the same backend.*"):
        concat_tensors(x, y)


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_container_mix_numpy_torch_v2():
    """
    Test error handling with mixed tensor types across containers.
    """
    import torch

    x = [np.array(1.0), np.array(2.0), np.array(3.0)]
    y = [np.array(4.0), np.array(5.0), torch.tensor(6.0)]
    with pytest.raises(TypeError, match=r".*must be from the same backend.*"):
        concat_tensors(x, y)
