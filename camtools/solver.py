"""
Functions for solving problems related to camera calibration and 3D geometry.
"""

import numpy as np
from typing import Tuple
from jaxtyping import Float
import open3d as o3d
from . import sanity


def line_intersection_3d(
    src_points: Float[np.ndarray, "n 3"],
    dst_points: Float[np.ndarray, "n 3"],
) -> Float[np.ndarray, "3"]:
    """
    Estimate 3D intersection of lines with least squares.

    Args:
        src_points: (N, 3) matrix containing starting point of N lines
        dst_points: (N, 3) matrix containing end point of N lines

    Returns:
        Estimated intersection point.

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


def closest_points_of_line_pair(
    src_o: Float[np.ndarray, "3"],
    src_d: Float[np.ndarray, "3"],
    dst_o: Float[np.ndarray, "3"],
    dst_d: Float[np.ndarray, "3"],
) -> Tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]:
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
        Tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]:
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


def closest_points_of_line_pairs(
    src_os: Float[np.ndarray, "n 3"],
    src_ds: Float[np.ndarray, "n 3"],
    dst_os: Float[np.ndarray, "n 3"],
    dst_ds: Float[np.ndarray, "n 3"],
) -> Tuple[Float[np.ndarray, "n 3"], Float[np.ndarray, "n 3"]]:
    """
    Find the closest points of two lines. The distance between the closest
    points is the shortest distance between the two lines.

    Args:
        src_os: (N, 3), origin of the src line
        src_ds: (N, 3), direction of the src line
        dst_os: (N, 3), origin of the dst line
        dst_ds: (N, 3), direction of the dst line

    Returns:
        Tuple[Float[np.ndarray, "n 3"], Float[np.ndarray, "n 3"]]:
            - src_ps: (N, 3), closest point of the src line
            - dst_ps: (N, 3), closest point of the dst line
    """
    sanity.assert_shape_nx3(src_os, "src_os")
    sanity.assert_shape_nx3(src_ds, "src_ds")
    sanity.assert_shape_nx3(dst_os, "dst_os")
    sanity.assert_shape_nx3(dst_ds, "dst_ds")

    # Normalize direction vectors.
    src_ds = src_ds / np.linalg.norm(src_ds, axis=1, keepdims=True)
    dst_ds = dst_ds / np.linalg.norm(dst_ds, axis=1, keepdims=True)

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
    mid_ds = np.cross(src_ds, dst_ds)
    mid_ds = mid_ds / np.linalg.norm(mid_ds, axis=1, keepdims=True)

    lhs = np.stack((src_ds, -dst_ds, mid_ds), axis=-1)
    rhs = dst_os - src_os
    results = np.linalg.solve(lhs, rhs)
    src_ts, dst_ts, mid_ts = results[:, 0], results[:, 1], results[:, 2]
    src_ps = src_os + src_ts.reshape((-1, 1)) * src_ds
    dst_ps = dst_os + dst_ts.reshape((-1, 1)) * dst_ds

    return src_ps, dst_ps


def point_plane_distance_three_points(
    point: Float[np.ndarray, "3"],
    plane_points: Float[np.ndarray, "3 3"],
) -> float:
    """
    Compute the distance between a point and a plane defined by three points.

    Args:
        point: (3,), point
        plane_points: (3, 3), three points on the plane

    Returns:
        Distance between the point and the plane.
    """
    if isinstance(point, list):
        point = np.array(point)
    if isinstance(plane_points, list):
        plane_points = np.array(plane_points)

    sanity.assert_shape_3(point, name="point")
    sanity.assert_shape_3x3(plane_points, name="plane_points")

    plane_a, plane_b, plane_c = plane_points

    # Compute the normal vector of the plane.
    plane_ab = plane_b - plane_a
    plane_ac = plane_c - plane_a
    plane_n = np.cross(plane_ab, plane_ac)
    plane_n = plane_n / np.linalg.norm(plane_n)

    # Compute the distance between the point and the plane.
    # Ref: https://mathworld.wolfram.com/Point-PlaneDistance.html
    distance = np.abs(np.dot(plane_n, point - plane_a))
    return distance


def points_to_mesh_distances(
    points: Float[np.ndarray, "n 3"],
    mesh: o3d.geometry.TriangleMesh,
) -> Float[np.ndarray, "n"]:
    """
    Compute the distance from points to a mesh surface.

    Args:
        points (np.ndarray): Array of points with shape (N, 3).
        mesh (o3d.geometry.TriangleMesh): The input mesh.

    Returns:
        Array of distances with shape (N,).
    """
    if not points.ndim == 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N, 3), but got {points.shape}.")
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_distance(points_tensor)
    return distances.numpy()
