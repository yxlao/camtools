"""
Functions for creating camera frustums and related visualizations.
"""

import open3d as o3d
import numpy as np
from typing import List, Optional, Dict, Tuple
from jaxtyping import Float
from . import convert
from . import sanity
from . import solver


def create_camera_frustums(
    Ks: Optional[List[Float[np.ndarray, "3 3"]]],
    Ts: List[Float[np.ndarray, "4 4"]],
    image_whs: Optional[List[List[int]]] = None,
    size: float = 0.1,
    color: Tuple[float, float, float] = (0, 0, 1),
    highlight_color_map: Optional[Dict[int, Tuple[float, float, float]]] = None,
    center_line: bool = True,
    center_line_color: Tuple[float, float, float] = (1, 0, 0),
    up_triangle: bool = True,
    center_ray: bool = False,
) -> o3d.geometry.LineSet:
    """
    Create Open3D line sets representing camera frustums.

    Args:
        Ks: Optional list of 3x3 intrinsic matrices. If None, uses default intrinsics.
        Ts: List of 4x4 extrinsic matrices (world-to-camera).
        image_whs: Optional list of [width, height] pairs for each image. If None,
            dimensions are inferred from intrinsics.
        size: Distance from camera center to image plane in world coordinates.
        color: RGB color tuple for frustum lines.
        highlight_color_map: Optional dict mapping camera indices to highlight colors.
            Supports negative indexing (e.g., {0: [0,1,0], -1: [1,0,0]}).
        center_line: If True, draws line connecting camera centers.
        center_line_color: RGB color tuple for center line.
        up_triangle: If True, draws triangle indicating camera up direction.
        center_ray: If True, draws ray from camera center to image center.

    Returns:
        o3d.geometry.LineSet: Open3D line set containing all frustum geometry.
    """
    if Ks is None:
        cx = 320
        cy = 240
        fx = 320
        fy = 320
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Ks = [K for _ in range(len(Ts))]
    if len(Ts) != len(Ks):
        raise ValueError(f"len(Ts) != len(Ks): {len(Ts)} != {len(Ks)}")
    for K in Ks:
        sanity.assert_K(K)
    for T in Ts:
        sanity.assert_T(T)

    if image_whs is None:
        image_whs = []
        for K in Ks:
            w = int((K[0, 2] + 0.5) * 2)
            h = int((K[1, 2] + 0.5) * 2)
            image_whs.append([w, h])
    else:
        if len(image_whs) != len(Ts):
            raise ValueError(
                f"len(image_whs) != len(Ts): {len(image_whs)} != {len(Ts)}"
            )
        for image_wh in image_whs:
            # Must be 2D.
            sanity.assert_shape_ndim(image_wh, nd=2)
            # Must be integer.
            w, h = image_wh
            if not isinstance(w, (int, np.integer)) or not isinstance(
                h, (int, np.integer)
            ):
                raise ValueError(f"image_wh must be integer, but got {image_wh}.")

    # Wrap the highlight_color_map dimensions.
    if highlight_color_map is not None:
        max_dim = len(Ts)
        highlight_color_map = {
            _wrap_dim(dim, max_dim, inclusive=False): color
            for dim, color in highlight_color_map.items()
        }

    # Draw camera frustums.
    ls = o3d.geometry.LineSet()
    for index, (T, K, image_wh) in enumerate(zip(Ts, Ks, image_whs)):
        if highlight_color_map is not None and index in highlight_color_map:
            frustum_color = highlight_color_map[index]
        else:
            frustum_color = color
        frustum = _create_camera_frustum(
            K,
            T,
            image_wh=image_wh,
            size=size,
            color=frustum_color,
            up_triangle=up_triangle,
            center_ray=center_ray,
        )
        ls += frustum

    # Draw camera center lines.
    if len(Ts) > 1 and center_line:
        # TODO: fix open3d linese += when one of the line set is empty, the
        # color will be missing,
        center_line = create_camera_center_line(Ts, color=center_line_color)
        ls += center_line

    return ls


def create_camera_frustum_with_Ts(
    Ts: List[Float[np.ndarray, "4 4"]],
    image_whs: Optional[List[List[int]]] = None,
    size: float = 0.1,
    color: Tuple[float, float, float] = (0, 0, 1),
    highlight_color_map: Optional[Dict[int, Tuple[float, float, float]]] = None,
    center_line: bool = True,
    center_line_color: Tuple[float, float, float] = (1, 0, 0),
    up_triangle: bool = True,
    center_ray: bool = False,
) -> o3d.geometry.LineSet:
    """
    Create camera frustums using only extrinsic matrices.

    Args:
        Ts: List of 4x4 extrinsic matrices (world-to-camera).
        image_whs: Optional list of [width, height] pairs for each image.
        size: Distance from camera center to image plane in world coordinates.
        color: RGB color tuple for frustum lines.
        highlight_color_map: Optional dict mapping camera indices to highlight colors.
        center_line: If True, draws line connecting camera centers.
        center_line_color: RGB color tuple for center line.
        up_triangle: If True, draws triangle indicating camera up direction.
        center_ray: If True, draws ray from camera center to image center.

    Returns:
        o3d.geometry.LineSet: Open3D line set containing all frustum geometry.
    """
    return create_camera_frustums(
        Ks=None,
        Ts=Ts,
        image_whs=image_whs,
        size=size,
        color=color,
        highlight_color_map=highlight_color_map,
        center_line=center_line,
        center_line_color=center_line_color,
        up_triangle=up_triangle,
        center_ray=center_ray,
    )


def create_camera_center_line(
    Ts: List[Float[np.ndarray, "4 4"]],
    color: Tuple[float, float, float] = (1, 0, 0),
) -> o3d.geometry.LineSet:
    """
    Create a line connecting camera centers.

    Args:
        Ts: List of 4x4 extrinsic matrices (world-to-camera).
        color: RGB color tuple for the center line.

    Returns:
        o3d.geometry.LineSet: Open3D line set connecting camera centers.
    """
    num_nodes = len(Ts)
    camera_centers = [convert.T_to_C(T) for T in Ts]

    ls = o3d.geometry.LineSet()
    lines = [[x, x + 1] for x in range(num_nodes - 1)]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(camera_centers)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls


def _create_camera_frustum(
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
    image_wh: List[int],
    size: float,
    color: Tuple[float, float, float],
    up_triangle: bool,
    center_ray: bool,
) -> o3d.geometry.LineSet:
    """
    Create a single camera frustum as an Open3D line set.

    Args:
        K: 3x3 intrinsic matrix.
        T: 4x4 extrinsic matrix (world-to-camera).
        image_wh: Image dimensions [width, height].
        size: Distance from camera center to image plane in world coordinates.
        color: RGB color tuple for frustum lines.
        up_triangle: If True, draws triangle indicating camera up direction.
        center_ray: If True, draws ray from camera center to image center.

    Returns:
        o3d.geometry.LineSet: Open3D line set representing the camera frustum.
    """
    T, K, color = np.asarray(T), np.asarray(K), np.asarray(color)
    sanity.assert_T(T)
    sanity.assert_K(K)
    sanity.assert_shape_3(color, "color")

    w, h = image_wh
    if not isinstance(w, (int, np.integer)) or not isinstance(h, (int, np.integer)):
        raise ValueError(f"image_wh must be integer, but got {image_wh}.")

    R, _ = convert.T_to_R_t(T)
    C = convert.T_to_C(T)

    # Compute distance of camera plane to origin.
    camera_plane_points_2d_homo = np.array(
        [
            [0, 0, 1],
            [w - 1, 0, 1],
            [0, h - 1, 1],
        ]
    )
    camera_plane_points_3d = (np.linalg.inv(K) @ camera_plane_points_2d_homo.T).T
    camera_plane_dist = solver.point_plane_distance_three_points(
        [0, 0, 0], camera_plane_points_3d
    )

    def points_2d_to_3d_world(points_2d):
        """
        Convert 2D points to 2D points in world coordinates.
        The points will be normalized by camera_plane_dist and scaled by size.
        """
        # Convert to homo coords.
        points_2d_homo = np.hstack((points_2d, np.ones((len(points_2d), 1))))
        # Camera space in world scale.
        points_3d_cam = (np.linalg.inv(K) @ points_2d_homo.T).T
        # Normalize to have distance 1 and scale by size.
        points_3d_cam = points_3d_cam / camera_plane_dist * size
        # Transform to world space.
        points_3d_world = (np.linalg.inv(R) @ points_3d_cam.T).T + C
        return points_3d_world

    # Camera frustum line set.
    points_2d = np.array(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ]
    )
    points_3d = points_2d_to_3d_world(points_2d)
    points = np.vstack((C, points_3d))
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ]
    )
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)

    if up_triangle:
        up_gap = 0.1 * h
        up_height = 0.5 * h
        up_points_2d = np.array(
            [
                [up_gap, -up_gap],
                [w - up_gap, -up_gap],
                [w / 2, -up_gap - up_height],
            ]
        )
        up_points = points_2d_to_3d_world(up_points_2d)
        up_lines = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ]
        )
        up_ls = o3d.geometry.LineSet()
        up_ls.points = o3d.utility.Vector3dVector(up_points)
        up_ls.lines = o3d.utility.Vector2iVector(up_lines)
        up_ls.paint_uniform_color(color)
        ls += up_ls

    if center_ray:
        center_px_2d = np.array([[(w - 1) / 2, (h - 1) / 2]])
        center_px_3d = points_2d_to_3d_world(center_px_2d)
        center_ray_points = np.vstack((C, center_px_3d))
        center_ray_lines = np.array([[0, 1]])
        center_ray_ls = o3d.geometry.LineSet()
        center_ray_ls.points = o3d.utility.Vector3dVector(center_ray_points)
        center_ray_ls.lines = o3d.utility.Vector2iVector(center_ray_lines)
        center_ray_ls.paint_uniform_color(color)
        ls += center_ray_ls

    return ls


def _wrap_dim(
    dim: int,
    max_dim: int,
    inclusive: bool = False,
) -> int:
    """
    Wrap a dimension index to be within valid bounds.

    Args:
        dim: The dimension index to wrap.
        max_dim: Maximum dimension size.
        inclusive: If True, max_dim is included in valid range.

    Returns:
        int: Wrapped dimension index within valid range.

    Raises:
        ValueError: If max_dim is <= 0 or dim is out of bounds.
    """
    if max_dim <= 0:
        raise ValueError(f"max_dim {max_dim} must be > 0.")
    min = -max_dim
    max = max_dim if inclusive else max_dim - 1

    if dim < min or dim > max:
        raise ValueError(
            f"Index out-of-range: dim == {dim}, "
            f"but it must satisfy {min} <= dim <= {max}."
        )
    if dim < 0:
        dim += max_dim
    return dim
