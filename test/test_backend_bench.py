import numpy as np
import camtools as ct
import pytest
from jaxtyping import Float

from camtools.backend import Tensor, ivy, is_torch_available, torch

_workload_repeat = 100


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_torch_to_numpy_manual(benchmark):

    def concat(
        x: Float[Tensor, "..."],
        y: Float[Tensor, "..."],
    ):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        for _ in range(_workload_repeat):
            np.concatenate([x, y], axis=0)
        return np.concatenate([x, y], axis=0)

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = benchmark(concat, x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.skipif(not is_torch_available(), reason="Torch is not available")
def test_concat_torch_to_numpy_auto(benchmark):

    @ct.backend.tensor_to_numpy_backend
    @ct.backend.tensor_type_check
    def concat(
        x: Float[Tensor, "..."],
        y: Float[Tensor, "..."],
    ):
        for _ in range(_workload_repeat):
            np.concatenate([x, y], axis=0)
        return np.concatenate([x, y], axis=0)

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = benchmark(concat, x, y)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
