"""
Functions for transforming points in 3D space.
"""

import numpy as np

from . import sanity
from . import convert


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

    points = convert.to_homo(points)
    points_transformed = points @ transform_mat.T  # (mat @ points.T).T
    points_transformed = convert.from_homo(points_transformed)

    return points_transformed
