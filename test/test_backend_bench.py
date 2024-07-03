import numpy as np
import camtools as ct
import pytest
from jaxtyping import Float

from camtools.backend import Tensor, ivy, is_torch_available, torch


def test_concat_numpy_native(benchmark):

    def concat(
        x: Float[Tensor, "..."],
        y: Float[Tensor, "..."],
    ):
        return np.concatenate([x, y], axis=0)

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    result = benchmark(concat, x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_torch_to_numpy(benchmark):

    @ct.backend.tensor_numpy_backend
    def concat(
        x: Float[Tensor, "..."],
        y: Float[Tensor, "..."],
    ):
        return np.concatenate([x, y], axis=0)

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = benchmark(concat, x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
