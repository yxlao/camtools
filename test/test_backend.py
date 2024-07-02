import numpy as np
import pytest
import camtools as ct
import ivy
from jaxtyping import Float
from camtools.typing import Tensor
from typing import Dict, Set


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


@ct.backend.with_auto_backend
@ct.typing.check_shape_and_dtype
def concat_tensor_set(x: Set[Float[Tensor, "..."]]):
    return ivy.concat(list(x), axis=0)


@ct.backend.with_auto_backend
@ct.typing.check_shape_and_dtype
def sum_tensor_dict(x: Dict[str, Float[Tensor, "..."]]):
    return ivy.add_n(list(x.values()))


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


def test_concat_tensor_set_numpy():
    """
    Test handling of sets containing numpy tensors.
    """
    x = {np.array(1.0), np.array(2.0), np.array(3.0)}
    result = concat_tensor_set(x)
    assert isinstance(result, np.ndarray)
    # Since sets are unordered, we can't predict the order of concatenation
    assert np.sort(result) == np.sort(np.array([1.0, 2.0, 3.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_tensor_set_torch():
    """
    Test handling of sets containing torch tensors.
    """
    import torch

    x = {torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)}
    with ct.backend.ScopedBackend("torch"):
        result = concat_tensor_set(x)
    assert isinstance(result, torch.Tensor)
    # Since sets are unordered, we can't predict the order of concatenation
    assert torch.sort(result).values.equal(
        torch.sort(torch.tensor([1.0, 2.0, 3.0])).values
    )


def test_sum_tensor_dict_numpy():
    """
    Test handling of dictionaries with numpy tensors as values.
    """
    x = {"a": np.array(1.0), "b": np.array(2.0), "c": np.array(3.0)}
    result = sum_tensor_dict(x)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0 + 2.0 + 3.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_sum_tensor_dict_torch():
    """
    Test handling of dictionaries with torch tensors as values.
    """
    import torch

    x = {"a": torch.tensor(1.0), "b": torch.tensor(2.0), "c": torch.tensor(3.0)}
    with ct.backend.ScopedBackend("torch"):
        result = sum_tensor_dict(x)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0 + 2.0 + 3.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_mixed_backends_in_containers_error():
    """
    Test error handling with mixed tensor types (NumPy and Torch) within the same or different containers.
    """
    import torch

    x = np.array([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = {x, y}  # Set containing different backend tensors
    w = {"a": x, "b": y}  # Dictionary containing different backend tensors

    with pytest.raises(TypeError, match=r".*must be from the same backend.*"):
        concat_tensor_set(z)
    with pytest.raises(TypeError, match=r".*must be from the same backend.*"):
        sum_tensor_dict(w)


def test_mix_numpy_with_list_in_set_numpy():
    """
    Test handling of mixing NumPy arrays with lists in a set.
    """
    x = {np.array([1.0, 2.0]), [3.0, 4.0]}
    result = concat_tensor_set(x)
    assert isinstance(result, np.ndarray)
    # Result ordering may vary
    assert np.allclose(np.sort(result), np.sort(np.array([1.0, 2.0, 3.0, 4.0])))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_mix_torch_with_list_in_set_torch():
    """
    Test handling of mixing Torch tensors with lists in a set.
    """
    import torch

    x = {torch.tensor([1.0, 2.0]), [3.0, 4.0]}
    with ct.backend.ScopedBackend("torch"):
        result = concat_tensor_set(x)
    assert isinstance(result, torch.Tensor)
    # Result ordering may vary
    assert torch.allclose(
        torch.sort(result).values, torch.sort(torch.tensor([1.0, 2.0, 3.0, 4.0])).values
    )


def test_mix_numpy_with_list_in_map_numpy():
    """
    Test handling of mixing NumPy arrays with lists in a dictionary.
    """
    x = {"a": np.array([1.0, 2.0]), "b": [3.0, 4.0]}
    result = sum_tensor_dict(x)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0 + 2.0 + 3.0 + 4.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_mix_torch_with_list_in_map_torch():
    """
    Test handling of mixing Torch tensors with lists in a dictionary.
    """
    import torch

    x = {"a": torch.tensor([1.0, 2.0]), "b": [3.0, 4.0]}
    with ct.backend.ScopedBackend("torch"):
        result = sum_tensor_dict(x)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0 + 2.0 + 3.0 + 4.0]))
