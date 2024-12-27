from typing import Optional, Union, Dict
import open3d as o3d
import numpy as np
from . import convert
from . import sanity
from . import solver
from jaxtyping import Float, Int, Bool


def create_camera_frustums(
    Ks: Optional[Float[np.ndarray, "n 3 3"]],
    Ts: Float[np.ndarray, "n 4 4"],
    image_whs: Optional[Int[np.ndarray, "n 2"]] = None,
    size: float = 0.1,
    color: Float[np.ndarray, "3"] = (0, 0, 1),
    highlight_color_map: Optional[Dict[int, Float[np.ndarray, "3"]]] = None,
    center_line: bool = True,
    center_line_color: Float[np.ndarray, "3"] = (1, 0, 0),
    up_triangle: bool = True,
    center_ray: bool = False,
) -> o3d.geometry.LineSet:
    """
    Create camera frustums in lineset.

    Args:
        Ks: Camera intrinsics matrices. You can set Ks to None if
            the intrinsics are not available. In this case, a dummy intrinsics
            matrix will be used.
        Ts: Camera extrinsics matrices.
        image_whs: Image width and height. If None, the image width and
            height are determined from the camera intrinsics by assuming that
            the camera offset is exactly at the center of the image.
        size: Distance from the camera center to image plane in world coordinates.
        color: Color of the camera frustums.
        highlight_color_map: A map of camera_index to color, specifying the
            colors of the highlighted cameras. Index wrapping is supported.
            For example, to highlight the start and stop cameras, use:
            highlight_color_map = {0: [0, 1, 0], -1: [1, 0, 0]}. If None, no
            camera is highlighted.
        center_line: If True, the camera center line will be drawn.
        center_line_color: Color of the camera center line.
        up_triangle: If True, the up triangle will be drawn.
        center_ray: If True, the ray from camera center to the center pixel in
            the image plane will be drawn.

    Returns:
        An Open3D lineset containing all the camera frustums.
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
    Ts: Float[np.ndarray, "n 4 4"],
    image_whs: Optional[Int[np.ndarray, "n 2"]] = None,
    size: float = 0.1,
    color: Float[np.ndarray, "3"] = (0, 0, 1),
    highlight_color_map: Optional[Dict[int, Float[np.ndarray, "3"]]] = None,
    center_line: bool = True,
    center_line_color: Float[np.ndarray, "3"] = (1, 0, 0),
    up_triangle: bool = True,
    center_ray: bool = False,
) -> o3d.geometry.LineSet:
    """
    Create camera frustums using only camera extrinsics matrices.

    A convenience wrapper around create_camera_frustums() that uses default
    camera intrinsics when none are available.

    Args:
        Ts: Camera extrinsics matrices.
        image_whs: Image width and height. If None, the image width and
            height are determined from default camera intrinsics.
        size: Distance from the camera center to image plane in world coordinates.
        color: Color of the camera frustums.
        highlight_color_map: A map of camera_index to color, specifying the
            colors of the highlighted cameras. Index wrapping is supported.
            For example, to highlight the start and stop cameras, use:
            highlight_color_map = {0: [0, 1, 0], -1: [1, 0, 0]}. If None, no
            camera is highlighted.
        center_line: If True, the camera center line will be drawn.
        center_line_color: Color of the camera center line.
        up_triangle: If True, the up triangle will be drawn.
        center_ray: If True, the ray from camera center to the center pixel in
            the image plane will be drawn.

    Returns:
        An Open3D lineset containing all the camera frustums.
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
    Ts: Float[np.ndarray, "n 4 4"],
    color: Float[np.ndarray, "3"] = np.array([1, 0, 0]),
) -> o3d.geometry.LineSet:
    """
    Create a line connecting the centers of consecutive cameras.

    Creates an Open3D LineSet that draws lines between the camera centers,
    useful for visualizing camera paths or trajectories.

    Args:
        Ts: Camera extrinsics matrices.
        color: RGB color for the center lines.

    Returns:
        An Open3D LineSet containing lines connecting consecutive camera centers.
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
    image_wh: Int[np.ndarray, "2"],
    size: float,
    color: Float[np.ndarray, "3"],
    up_triangle: bool,
    center_ray: bool,
) -> o3d.geometry.LineSet:
    """
    Create a single camera frustum visualization.

    Args:
        K: Camera intrinsics matrix.
        T: Camera extrinsics matrix.
        image_wh: Image width and height.
        size: Distance from the camera center to image plane in world coordinates.
        color: RGB color for the frustum.
        up_triangle: If True, draws a triangle above the frustum to indicate up direction.
        center_ray: If True, draws a ray from camera center to image center.

    Returns:
        An Open3D LineSet representing the camera frustum.
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


def _wrap_dim(dim: int, max_dim: int, inclusive: bool = False) -> int:
    """
    Wraps a dimension index into valid range, supporting negative indexing.

    Takes a dimension index and wraps it to be within [-max_dim, max_dim) or
    [-max_dim, max_dim] if inclusive=True. Negative indices are converted to
    their positive equivalents.

    Args:
        dim: The dimension index to wrap.
        max_dim: The maximum dimension value (must be > 0).
        inclusive: If True, max_dim is included in valid range.

    Returns:
        The wrapped dimension index.

    Raises:
        ValueError: If max_dim <= 0 or if dim is out of valid range.
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
