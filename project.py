import numpy as np
import torch
from . import sanity
from . import convert


def homo_project(points, mat):
    sanity.assert_shape_nx3(points, name="points")
    sanity.assert_shape_4x4(mat, name="mat")
    sanity.assert_same_device(points, mat)

    N = len(points)
    if torch.is_tensor(mat):
        ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
        points_homo = torch.hstack((points, ones))
    else:
        ones = np.ones((N, 1))
        points_homo = np.hstack((points, ones))

    # (mat @ points_homo.T).T
    points_out = points_homo @ mat.T
    points_out = points_out[:, :3] / points_out[:, 3:]
    return points_out


def points_to_pixel(points, K, T):
    """
    Project points in world coordinates to pixel coordinates.

    Example usage:
        pixels = ct.project.points_to_pixel(points, K, T)

        cols = pixels[:, 0]  # cols, width, x, top-left to top-right
        rows = pixels[:, 1]  # rows, height, y, top-left to bottom-left
        cols = np.round(cols).astype(np.int32)
        rows = np.round(rows).astype(np.int32)
        cols[cols >= width] = width - 1
        cols[cols < 0] = 0
        rows[rows >= height] = height - 1
        rows[rows < 0] = 0

    Args:
        K: (3, 3) array, camera intrinsic matrix.
        T: (4, 4) array, camera extrinsic matrix, [R | t] with [0, 0, 0, 1]
           below.
        points: (N, 3) array, 3D points in world coordinates.

    Return:
        (N, 2) array, representing [cols, rows] by each column. N is the number
        of points, which is not related to the image height and width.
    """
    sanity.assert_K(K)
    sanity.assert_T(T)
    sanity.assert_shape_nx3(points, name="points")

    W2P = convert.K_T_to_W2P(K, T)

    # points_homo: (N, 4)
    N = len(points)
    if torch.is_tensor(points):
        ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
        points_homo = torch.hstack((points, ones))
    else:
        ones = np.ones((N, 1))
        points_homo = np.hstack((points, ones))

    # points_out: (N, 4)
    # points_out = (W2P @ points_homo.T).T
    #              = points_homo @ W2P.T
    points_out = points_homo @ W2P.T

    # points_out: (N, 3)
    # points_out disgard last column
    points_out = points_out[:, :3]

    # points_out: (N, 2)
    # points_out convert homo to regular
    points_out = points_out[:, :2] / points_out[:, 2:]

    return points_out
