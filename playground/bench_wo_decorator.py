import numpy as np
from jaxtyping import Float

import camtools as ct
import cProfile
from camtools.backend import Tensor, torch

_array_repeat = 1000


def dot(
    x: Float[Tensor, "..."],
    y: Float[Tensor, "..."],
):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    x = np.repeat(x, _array_repeat)
    y = np.repeat(y, _array_repeat)
    return np.dot(x, y)


def run_without_decorator():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = dot(x, y)
    assert result == 32.0 * _array_repeat


def main():
    profiler = cProfile.Profile()

    profiler.enable()
    run_without_decorator()
    profiler.disable()
    profiler.dump_stats("wo_decorator.prof")


if __name__ == "__main__":
    main()
