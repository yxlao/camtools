import numpy as np
import torch

from camtools import sanity


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
        raise ValueError(f"src_points must be (N, 3), but got {src_points.shape}.")
    if dst_points.ndim != 2 or dst_points.shape[1] != 3:
        raise ValueError(f"dst_points must be (N, 3), but got {dst_points.shape}.")

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
    s = np.array(
        [
            [sxx, sxy, sxz],
            [sxy, syy, syz],
            [sxz, syz, szz],
        ]
    )

    # RHS
    # fmt: off
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
    # fmt: on

    return np.linalg.solve(s, c)


def closest_points_of_line_pair(src_o, src_d, dst_o, dst_d):
    """
    Find the closest points of two lines. The distance between the closest
    points is the shortest distance between the two lines. Used the batched
    version closest_points_of_line_pairs() when possible.

    Args:
        src_o: (3,), origin of the src line
        src_d: (3,), direction of the src line
        dst_o: (3,), origin of the dst line
        dst_d: (3,), direction of the dst line

    Returns:
        (src_p, dst_p), where
        - src_p: (3,), closest point of the src line
        - dst_p: (3,), closest point of the dst line
    """
    sanity.assert_shape_3(src_o, "src_o")
    sanity.assert_shape_3(src_d, "src_d")
    sanity.assert_shape_3(dst_o, "dst_o")
    sanity.assert_shape_3(dst_d, "dst_d")

    src_ps, dst_ps = closest_points_of_line_pairs(
        src_o.reshape((1, 3)),
        src_d.reshape((1, 3)),
        dst_o.reshape((1, 3)),
        dst_d.reshape((1, 3)),
    )

    return src_ps[0], dst_ps[0]


def closest_points_of_line_pairs(src_os, src_ds, dst_os, dst_ds):
    """
    Find the closest points of two lines. The distance between the closest
    points is the shortest distance between the two lines.

    Args:
        src_os: (N, 3), origin of the src line
        src_ds: (N, 3), direction of the src line
        dst_os: (N, 3), origin of the dst line
        dst_ds: (N, 3), direction of the dst line

    Returns:
        (src_ps, dst_s), where
        - src_ps: (N, 3), closest point of the src line
        - dst_ps: (N, 3), closest point of the dst line
    """
    sanity.assert_shape_nx3(src_os, "src_os")
    sanity.assert_shape_nx3(src_ds, "src_ds")
    sanity.assert_shape_nx3(dst_os, "dst_os")
    sanity.assert_shape_nx3(dst_ds, "dst_ds")

    is_torch = torch.is_tensor(src_ds) and torch.is_tensor(dst_ds)
    cross = torch.cross if is_torch else np.cross
    norm = torch.linalg.norm if is_torch else np.linalg.norm
    solve = torch.linalg.solve if is_torch else np.linalg.solve
    stack = torch.stack if is_torch else np.stack

    # Normalize direction vectors.
    src_ds = src_ds / norm(src_ds, axis=1, keepdims=True)
    dst_ds = dst_ds / norm(dst_ds, axis=1, keepdims=True)

    # Find the closest points of the two lines.
    # - src_p = src_o + src_t * src_d is the closest point in src line.
    # - dst_p = dst_o + dst_t * dst_d is the closest point in dst line.
    # - src_p + mid_t * mid_d == dst_d connects the two points, this expands to:
    #   src_o + src_t * src_d + mid_t * mid_d == dst_o + dst_t * dst_d
    # - mid_d = cross(src_d, dst_d).
    #
    # Solve the following linear system:
    #   src_t * src_d - dst_t * dst_d + mid_t * mid_d == dst_o - src_o
    #   ┌                  ┐ ┌       ┐   ┌       ┐   ┌       ┐
    #   │  │     │     │   │ │ src_t │   │   │   │   │   │   │
    #   │src_d -dst_d mid_d│ │ dst_t │ = │ dst_o │ - │ src_o │
    #   │  │     │     │   │ │ mid_t │   │   │   │   │   │   │
    #   └                  ┘ └       ┘   └       ┘   └       ┘
    mid_ds = cross(src_ds, dst_ds)
    mid_ds = mid_ds / norm(mid_ds, axis=1, keepdims=True)

    lhs = stack((src_ds, -dst_ds, mid_ds), axis=-1)
    rhs = dst_os - src_os
    results = solve(lhs, rhs)
    src_ts, dst_ts, mid_ts = results[:, 0], results[:, 1], results[:, 2]
    src_ps = src_os + src_ts.reshape((-1, 1)) * src_ds
    dst_ps = dst_os + dst_ts.reshape((-1, 1)) * dst_ds

    return src_ps, dst_ps
