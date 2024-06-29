"""
Test basic usage of ivy and its interaction with numpy and torch.
"""

import ivy
import numpy as np
import torch
import einops
import camtools as ct


def test_creation():

    @ct.backend.with_native_backend
    def creation():
        zeros = ivy.zeros([2, 3])
        return zeros

    tensor = creation()
    assert isinstance(tensor, np.ndarray)
