"""
Functions for transforming points in 3D space.
"""

import numpy as np
import torch
from . import sanity


def transform_points(points, transform_mat):
    """
    Transform points by a 4x4 matrix via homogenous coordinates projection.

    If the last row of the matrix is [0, 0, 0, 1], then the transform is
    assumed to be a rigid transform (rotation + translation).

    Args:
        points: (N, 3) array.
        mat: (4, 4) array, the transformation matrix.

    Returns:
        (N, 3) array, the transformed points.
    """
    if torch.is_tensor(points):
        return _transform_points_torch(points, transform_mat)

    sanity.assert_shape_nx3(points, name="points")
    sanity.assert_shape_4x4(transform_mat, name="mat")
    sanity.assert_same_device(points, transform_mat)

    if np.allclose(transform_mat[3, :3], [0, 0, 0]):
        # Faster method.
        R = transform_mat[:3, :3]
        t = transform_mat[:3, 3]
        points_rotated = points @ R.T
        points_transformed = points_rotated + t
        return points_transformed
    else:
        # Arbitrary transform.
        N = len(points)
        ones = np.ones((N, 1), dtype=points.dtype)
        points_homo = np.hstack((points, ones))
        # (mat @ points_homo.T).T
        points_transformed = points_homo @ transform_mat.T
        points_transformed = points_transformed[:, :3] / points_transformed[:, 3:]
        return points_transformed


def _transform_points_torch(points, transform_mat):
    """
    Torch version of the transform_points.
    """
    sanity.assert_shape_nx3(points, name="points")
    sanity.assert_shape_4x4(transform_mat, name="mat")
    sanity.assert_same_device(points, transform_mat)

    if torch.allclose(transform_mat[3, :3], [0, 0, 0]):
        # Faster method.
        R = transform_mat[:3, :3]
        t = transform_mat[:3, 3]
        points_rotated = points @ R.T
        points_transformed = points_rotated + t
        return points_transformed
    else:
        # Arbitrary transform.
        N = len(points)
        ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
        points_homo = torch.hstack((points, ones))

        points_out = points_homo @ transform_mat.T
        points_out = points_out[:, :3] / points_out[:, 3:]
        return points_out
