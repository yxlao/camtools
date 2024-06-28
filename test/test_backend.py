"""
Test basic usage of ivy and its interaction with numpy and torch.
"""

import ivy
import numpy as np
import torch
import einops
import camtools as ct


@ct.backend.with_native_backend
def creation():
    zeros = ivy.zeros([2, 3])
    return zeros


def test_creation():
    zeros = creation()
    import ipdb

    ipdb.set_trace()
    pass

    # # Default backend
    # zeros = ivy.zeros([2, 3])
    # assert zeros.backend == ivy.current_backend().backend
    # assert zeros.backend == "numpy"
    # assert zeros.dtype == ivy.float32
    # assert zeros.shape == (2, 3)
    # zeros = zeros.to_native()

    # import ipdb

    # ipdb.set_trace()
    # pass

    # # Explicit numpy
    # ivy.set_backend("numpy")
    # zeros = ivy.zeros([2, 3])
    # assert isinstance(zeros.data, np.ndarray)

    # pass


ivy.set_array_mode
