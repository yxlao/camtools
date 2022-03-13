import numpy as np


def line_intersection_3d(src_points=None, dst_points=None):
    """
    Estimate 3D intersection of lines with least squares.

    Args:
        src_points: (N, 3) matrix containing starting point of N lines
        dst_points: (N, 3) matrix containing end point of N lines

    Returns:
        intersection: (3,) Estimated intersection point.

    Ref:
        https://math.stackexchange.com/a/1762491/209055
    """
    if src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError(
            f"src_points must be (N, 3), but got {src_points.shape}.")
    if dst_points.ndim != 2 or dst_points.shape[1] != 3:
        raise ValueError(
            f"dst_points must be (N, 3), but got {dst_points.shape}.")

    dirs = dst_points - src_points
    dirs = dirs / np.linalg.norm(dirs, axis=1).reshape((-1, 1))

    nx = dirs[:, 0]
    ny = dirs[:, 1]
    nz = dirs[:, 2]

    # Coefficients
    sxx = np.sum(nx**2 - 1)
    syy = np.sum(ny**2 - 1)
    szz = np.sum(nz**2 - 1)
    sxy = np.sum(nx * ny)
    sxz = np.sum(nx * nz)
    syz = np.sum(ny * nz)
    s = np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz],
    ])

    # RHS
    # yapf: disable
    cx = np.sum(src_points[:, 0] * (nx**2 - 1) +
                src_points[:, 1] * nx * ny     +
                src_points[:, 2] * nx * nz)
    cy = np.sum(src_points[:, 0] * nx * ny     +
                src_points[:, 1] * (ny**2 - 1) +
                src_points[:, 2] * ny * nz)
    cz = np.sum(src_points[:, 0] * nx * nz     +
                src_points[:, 1] * ny * nz     +
                src_points[:, 2] * (nz**2 - 1))
    c = np.array([cx, cy, cz])
    # yapf: enable

    return np.linalg.solve(s, c)
