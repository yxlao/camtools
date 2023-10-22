"""
Functions for transforming points in 3D space.
"""

import numpy as np
import torch
from . import sanity


def transform_points(points, transform_mat):
    """
    Transform points by a 4x4 matrix via homogenous coordinates projection.

    Args:
        points: (N, 3) array.
        mat: (4, 4) array, the transformation matrix.

    Returns:
        (N, 3) array, the transformed points.
    """
    sanity.assert_shape_nx3(points, name="points")
    sanity.assert_shape_4x4(transform_mat, name="mat")
    sanity.assert_same_device(points, transform_mat)

    N = len(points)
    if torch.is_tensor(transform_mat):
        ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
        points_homo = torch.hstack((points, ones))
    else:
        ones = np.ones((N, 1))
        points_homo = np.hstack((points, ones))

    # (mat @ points_homo.T).T
    points_out = points_homo @ transform_mat.T
    points_out = points_out[:, :3] / points_out[:, 3:]
    return points_out
