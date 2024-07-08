import numpy as np
from jaxtyping import Float

import camtools as ct
from camtools.backend import Tensor, torch


@ct.backend.tensor_to_numpy_backend
def _dot(
    x: Float[Tensor, "..."],
    y: Float[Tensor, "..."],
):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    return np.dot(x, y)


def main():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = _dot(x, y)
    assert result == 32.0
    print("Success!")


if __name__ == "__main__":
    main()
