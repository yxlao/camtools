import numpy as np


def homo_project(mat, points):
    if mat.shape != (4, 4):
        raise ValueError(f"mat must be (4, 4), but got {mat.shape}.")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), but got {points.shape}.")

    N = len(points)
    points_homo = np.hstack((points, np.ones((N, 1))))

    # (mat @ points_homo.T).T
    points_out = points_homo @ mat.T
    points_out = points_out[:, :3] / points_out[:, 3:]
    return points_out


def world_to_pixel_with_world_mat(world_mat, points):
    """
    world_mat: consistent with the world_mat used in NeuS.
    points: must be (N, 3) array.
    """
    if world_mat.shape != (4, 4):
        raise ValueError(
            f"world_mat must be (4, 4), but got {world_mat.shape}.")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), but got {points.shape}.")

    # points_homo: (N, 4)
    N = len(points)
    points_homo = np.hstack((points, np.ones((N, 1))))

    # import ipdb
    # ipdb.set_trace()

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
