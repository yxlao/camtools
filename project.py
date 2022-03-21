import numpy as np
import torch
from . import sanity


def homo_project(mat, points):
    sanity.assert_shape_4x4(mat, name="mat")
    sanity.assert_shape_nx3(points, name="points")
    sanity.assert_same_device(mat, points)

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


def world_to_pixel_with_world_mat(world_mat, points):
    """
    Example usage:
        pixels = ct.project.world_to_pixel_with_world_mat(world_mat, vertices)
        cols = pixels[:, 0]  # cols, width, x, top-left to top-right
        rows = pixels[:, 1]  # rows, height, y, top-left to bottom-left
        cols = np.round(cols).astype(np.int32)
        rows = np.round(rows).astype(np.int32)
        cols[cols >= width] = width - 1
        cols[cols < 0] = 0
        rows[rows >= height] = height - 1
        rows[rows < 0] = 0

    Arguments:
        world_mat: (4, 4) array, world-to-pixel projection matrix. It is P with
            (0, 0, 0, 1) row below.
        points: (N, 3) array, 3D points.

    Return:
        (N, 2) array, representing [cols, rows] by each column.
    """
    sanity.assert_shape_4x4(world_mat, name="world_mat")
    sanity.assert_shape_nx3(points, name="points")

    # points_homo: (N, 4)
    N = len(points)
    if torch.is_tensor(points):
        ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)
        points_homo = torch.hstack((points, ones))
    else:
        ones = np.ones((N, 1))
        points_homo = np.hstack((points, ones))

    # points_out: (N, 4)
    # points_out = (world_mat @ points_homo.T).T
    #              = points_homo @ world_mat.T
    points_out = points_homo @ world_mat.T

    # points_out: (N, 3)
    # points_out disgard last column
    points_out = points_out[:, :3]

    # points_out: (N, 2)
    # points_out convert homo to regular
    points_out = points_out[:, :2] / points_out[:, 2:]

    return points_out
