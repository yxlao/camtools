import numpy as np
from jaxtyping import Float

import camtools as ct
import cProfile
from camtools.backend import Tensor, torch

_array_repeat = 1000


@ct.backend.tensor_to_numpy_backend
@ct.backend.tensor_type_check
def _dot_with_decorator(
    x: Float[Tensor, "..."],
    y: Float[Tensor, "..."],
):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    x = np.repeat(x, _array_repeat)
    y = np.repeat(y, _array_repeat)
    return np.dot(x, y)


def dot(
    x: Float[Tensor, "..."],
    y: Float[Tensor, "..."],
):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    x = np.repeat(x, _array_repeat)
    y = np.repeat(y, _array_repeat)
    return np.dot(x, y)


def run_with_decorator():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = _dot_with_decorator(x, y)
    assert result == 32.0 * _array_repeat


def run_without_decorator():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = dot(x, y)
    assert result == 32.0 * _array_repeat


def main():
    profiler = cProfile.Profile()

    # Profile run_with_decorator
    profiler.enable()
    run_with_decorator()
    profiler.disable()
    profiler.dump_stats("w_decorator.prof")

    # Profile run_without_decorator
    profiler.enable()
    run_without_decorator()
    profiler.disable()
    profiler.dump_stats("wo_decorator.prof")


if __name__ == "__main__":
    main()
