import numpy as np
from jaxtyping import Float

import camtools as ct
from camtools.backend import Tensor, torch


@ct.backend.tensor_to_numpy_backend
def _sum(
    x: Float[Tensor, "3"],
    y: Float[Tensor, "3"] = torch.tensor([2.0, 2.0, 2.0]),
    z: Float[Tensor, "3"] = torch.tensor([3.0, 3.0, 3.0]),
):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(z, np.ndarray)
    return x + y + z


def main():
    x = torch.tensor([1.0, 1.0, 1.0])
    y = torch.tensor([5.0, 5.0, 5.0])
    result = _sum(x, y=y)
    expected_result = np.array([9.0, 9.0, 9.0])
    assert np.allclose(result, expected_result)
    print("All tests passed!")


if __name__ == "__main__":
    main()
