"""
Functions for transforming points in 3D space.
"""

import numpy as np
from jaxtyping import Float

from . import sanity
from . import convert


def transform_points(
    points: Float[np.ndarray, "n 3"],
    transform_mat: Float[np.ndarray, "4 4"],
) -> Float[np.ndarray, "n 3"]:
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


def transform_point(
    point: Float[np.ndarray, "3"],
    transform_mat: Float[np.ndarray, "4 4"],
) -> Float[np.ndarray, "3"]:
    """
    Transform a single point by a 4x4 matrix via homogenous coordinates projection.

    Args:
        point: (3,) array.
        mat: (4, 4) array, the transformation matrix.

    Returns:
        (3,) array, the transformed point.
    """
    sanity.assert_shape_3(point, name="point")
    sanity.assert_shape_4x4(transform_mat, name="mat")

    point_transformed = transform_points(point[None], transform_mat)[0]

    return point_transformed
