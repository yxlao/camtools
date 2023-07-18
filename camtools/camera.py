import open3d as o3d
import numpy as np
from . import convert
from . import sanity


def create_camera_center_line(Ts, color=np.array([1, 0, 0])):
    num_nodes = len(Ts)
    camera_centers = [convert.T_to_C(T) for T in Ts]

    ls = o3d.geometry.LineSet()
    lines = [[x, x + 1] for x in range(num_nodes - 1)]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(camera_centers)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls


def create_camera_frame(T, size=0.1, color=[0, 0, 1]):
    R, t = T[:3, :3], T[:3, 3]

    C0 = convert.R_t_to_C(R, t).ravel()
    C1 = (
        C0 + R.T.dot(np.array([[-size], [-size], [3 * size]], dtype=np.float32)).ravel()
    )
    C2 = (
        C0 + R.T.dot(np.array([[-size], [+size], [3 * size]], dtype=np.float32)).ravel()
    )
    C3 = (
        C0 + R.T.dot(np.array([[+size], [+size], [3 * size]], dtype=np.float32)).ravel()
    )
    C4 = (
        C0 + R.T.dot(np.array([[+size], [-size], [3 * size]], dtype=np.float32)).ravel()
    )

    ls = o3d.geometry.LineSet()
    points = np.array([C0, C1, C2, C3, C4])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls


def create_camera_frames(
    Ts,
    size=0.1,
    color=[0, 0, 1],
    start_color=[0, 1, 0],
    end_color=[1, 0, 0],
    center_line=True,
    center_line_color=[1, 0, 0],
):
    camera_frames = o3d.geometry.LineSet()
    for index, T in enumerate(Ts):
        if index == 0:
            frame_color = start_color
        elif index == len(Ts) - 1:
            frame_color = end_color
        else:
            frame_color = color
        camera_frame = create_camera_frame(T, size=size, color=frame_color)
        camera_frames += camera_frame

    if len(Ts) > 1 and center_line:
        center_line = create_camera_center_line(Ts, color=center_line_color)
        camera_frames += center_line

    return camera_frames


def create_camera_center_ray(K, T, size=0.1, color=[0, 0, 1]):
    """
    K: 3x3
    T: 4x4

    Returns a linset of two points. The line starts the camera center and passes
    through the center of the image.
    """
    sanity.assert_T(T)
    sanity.assert_K(K)

    # Pick point at the center of the image
    # Assumes that the camera offset is exactly at the center of the image.
    col = K[0, 2]
    row = K[1, 2]
    points = np.array(
        [
            [col, row, 1],
        ]
    )

    # Transform to camera space
    points = (np.linalg.inv(K) @ points.T).T

    # Normalize to have 1 distance
    points = points / np.linalg.norm(points, axis=1, keepdims=True) * size

    # Transform to world space
    R, _ = convert.T_to_R_t(T)
    C = convert.T_to_C(T)
    points = (np.linalg.inv(R) @ points.T).T + C

    # Create line set
    points = np.vstack((C, points))
    lines = np.array(
        [
            [0, 1],
        ]
    )
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)

    return ls


def create_camera_center_rays(Ks, Ts, size=0.1, color=[0, 0, 1]):
    """
    K: 3x3
    T: 4x4

    Returns a linset of two points. The line starts the camera center and passes
    through the center of the image.
    """
    if len(Ts) != len(Ks):
        raise ValueError(f"len(Ts) != len(Ks)")

    camera_rays = o3d.geometry.LineSet()
    for T, K in zip(Ts, Ks):
        camera_rays += create_camera_center_ray(T, K, size=size, color=color)

    return camera_rays


def create_camera_ray_frame(K, T, size=0.1, color=[0, 0, 1]):
    """
    K: 3x3
    T: 4x4
    """
    T, K, color = np.asarray(T), np.asarray(K), np.asarray(color)
    sanity.assert_T(T)
    sanity.assert_K(K)
    sanity.assert_shape_3(color, "color")

    # Pick 4 corner points
    # Assumes that the camera offset is exactly at the center of the image.
    # The rays are plotted in the center of each corner pixel.
    w = (K[0, 2] + 0.5) * 2 - 1
    h = (K[1, 2] + 0.5) * 2 - 1
    points = np.array(
        [
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1],
        ]
    )

    # Transform to camera space
    points = (np.linalg.inv(K) @ points.T).T

    # Normalize to have 1 distance
    points = points / np.linalg.norm(points, axis=1, keepdims=True) * size

    # Transform to world space
    R, _ = convert.T_to_R_t(T)
    C = convert.T_to_C(T)
    points = (np.linalg.inv(R) @ points.T).T + C

    # Create line set
    points = np.vstack((C, points))
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

    return ls


def _wrap_dim(dim: int, max_dim: int, inclusive: bool = False) -> int:
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


def create_camera_ray_frames(
    Ks,
    Ts,
    size=0.1,
    color=[0, 0, 1],
    highlight_color_map=None,
    center_line=True,
    center_line_color=[1, 0, 0],
):
    """
    Args:
        Ks: List of 3x3 camera intrinsics matrices.
        Ts: List of 4x4 camera extrinsics matrices.
        size: Size of the camera camera frame.
        color: Color of the camera frame.
        highlight_color_map: A map of camera_index to color, specifying the
            colors of the highlighted cameras. Index wrapping is supported.
            For example, to highlight the start and stop cameras, use:
            highlight_color_map = {0: [0, 1, 0], -1: [1, 0, 0]}. If None, no
            camera is highlighted.
        center_line: If True, the camera center line will be drawn.
        center_line_color: Color of the camera center line.
    """
    assert len(Ts) == len(Ks)

    # Wrap the highlight_color_map dimensions.
    if highlight_color_map is not None:
        max_dim = len(Ts)
        highlight_color_map = {
            _wrap_dim(dim, max_dim, inclusive=False): color
            for dim, color in highlight_color_map.items()
        }

    # Draw camera frames.
    ls = o3d.geometry.LineSet()
    for index, (T, K) in enumerate(zip(Ts, Ks)):
        if highlight_color_map is not None and index in highlight_color_map:
            frame_color = highlight_color_map[index]
        else:
            frame_color = color
        camera_frame = create_camera_ray_frame(K, T, size=size, color=frame_color)
        ls += camera_frame

    # Draw camera center lines.
    if len(Ts) > 1 and center_line:
        # TODO: fix open3d linese += when one of the line set is empty, the
        # color will be missing,
        center_line = create_camera_center_line(Ts, color=center_line_color)
        ls += center_line

    return ls
