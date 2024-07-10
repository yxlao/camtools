import warnings

import numpy as np
import pytest
from jaxtyping import Float

import camtools as ct
from camtools.backend import Tensor, is_torch_available, ivy, torch


@pytest.fixture(autouse=True)
def ignore_ivy_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*Compositional function.*array_mode is set to False.*",
        category=UserWarning,
    )
    yield


# Define numpy arrays for testing
in_val_2d = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ],
    dtype=np.float64,
)
gt_out_val_2d = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)
in_val_3d = np.array(
    [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        [
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ],
    ],
    dtype=np.float64,
)
gt_out_val_3d = np.array(
    [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [0, 0, 0, 1],
        ],
        [
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
            [0, 0, 0, 1],
        ],
    ],
    dtype=np.float64,
)


def test_pad_0001_numpy(benchmark):

    def workload():
        out_val_2d = ct.convert.pad_0001(in_val_2d)
        out_val_3d = ct.convert.pad_0001(in_val_3d)
        return out_val_2d, out_val_3d

    out_val_2d, out_val_3d = benchmark(workload)
    assert np.allclose(out_val_2d, gt_out_val_2d)
    assert np.allclose(out_val_3d, gt_out_val_3d)


def test_pad_0001_torch(benchmark):

    in_val_2d_torch = torch.from_numpy(in_val_2d)
    in_val_3d_torch = torch.from_numpy(in_val_3d)

    def workload():
        out_val_2d = ct.convert.pad_0001(in_val_2d_torch)
        out_val_3d = ct.convert.pad_0001(in_val_3d_torch)
        return out_val_2d, out_val_3d

    out_val_2d, out_val_3d = benchmark(workload)
    assert torch.equal(out_val_2d, torch.from_numpy(gt_out_val_2d))
    assert torch.equal(out_val_3d, torch.from_numpy(gt_out_val_3d))


def test_pad_0001_classic_numpy(benchmark):

    def workload():
        out_val_2d = ct.convert.pad_0001_classic(in_val_2d)
        out_val_3d = ct.convert.pad_0001_classic(in_val_3d)
        return out_val_2d, out_val_3d

    out_val_2d, out_val_3d = benchmark(workload)
    assert np.allclose(out_val_2d, gt_out_val_2d)
    assert np.allclose(out_val_3d, gt_out_val_3d)


def test_pad_0001_classic_torch(benchmark):

    in_val_2d_torch = torch.from_numpy(in_val_2d)
    in_val_3d_torch = torch.from_numpy(in_val_3d)

    def workload():
        out_val_2d = ct.convert.pad_0001_classic(in_val_2d_torch)
        out_val_3d = ct.convert.pad_0001_classic(in_val_3d_torch)
        return out_val_2d, out_val_3d

    out_val_2d, out_val_3d = benchmark(workload)
    assert torch.equal(out_val_2d, torch.from_numpy(gt_out_val_2d))
    assert torch.equal(out_val_3d, torch.from_numpy(gt_out_val_3d))
