import open3d as o3d
import numpy as np
from . import convert


def get_camera_center_line(Ts, color=np.array([1, 0, 0])):
    num_nodes = len(Ts)
    camera_centers = [convert.T_to_C(T) for T in Ts]

    ls = o3d.geometry.LineSet()
    lines = [[x, x + 1] for x in range(num_nodes - 1)]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(camera_centers)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls

def get_camera_frame(T, size=0.1, color=[0, 0, 1]):

    R, t = T[:3, :3], T[:3, 3]

    C0 = convert.R_t_to_C(R, t).ravel()
    C1 = (C0 + R.T.dot(
        np.array([[-size], [-size], [3 * size]], dtype=np.float32)).ravel())
    C2 = (C0 + R.T.dot(
        np.array([[-size], [+size], [3 * size]], dtype=np.float32)).ravel())
    C3 = (C0 + R.T.dot(
        np.array([[+size], [+size], [3 * size]], dtype=np.float32)).ravel())
    C4 = (C0 + R.T.dot(
        np.array([[+size], [-size], [3 * size]], dtype=np.float32)).ravel())

    ls = o3d.geometry.LineSet()
    points = np.array([C0, C1, C2, C3, C4])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls

def get_camera_frames(Ts,
                      size=0.1,
                      color=[0, 0, 1],
                      center_line=True,
                      center_line_color=[1, 0, 0]):

    camera_frames = o3d.geometry.LineSet()
    for T in Ts:
        camera_frame = get_camera_frame(T, size=size, color=color)
        camera_frames += camera_frame

    if len(Ts) > 1  and center_line:
        center_line = get_camera_center_line(Ts, color=center_line_color)
        camera_frames += center_line

    return camera_frames


def get_camera_ray_frame(T, K, size=0.1, color=[0, 0, 1]):
    """
    T: 4x4
    K: 3x3
    """
    T, K, color = np.asarray(T), np.asarray(K), np.asarray(color)
    if T.shape != (4, 4):
        raise ValueError(
            f"T must has shape (4, 4), but got {T.shape}.")
    if not np.allclose(T[3, :], [0, 0, 0, 1]):
        raise ValueError(
            f"T must has [0, 0, 0, 1] the bottom line, but got {T}.")
    if K.shape != (3, 3):
        raise ValueError(
            f"K must has shape (3, 3), but got {K.shape}.")
    if color.shape != (3,):
        raise ValueError(
            f"color must has shape (3,), but got {color.shape}.")

    # Pick 4 corner points
    # Assumes that the camera offset is exactly at the center of the image.
    # The rays are plotted in the center of each corner pixel.
    w = (K[0, 2] + 0.5) * 2 - 1
    h = (K[1, 2] + 0.5) * 2 - 1
    points = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1],
    ])

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
    lines = np.array(([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)

    return ls

def get_camera_ray_frames(Ts,
                          Ks,
                          size=0.1,
                          color=[0, 0, 1],
                          center_line=True,
                          center_line_color=[1, 0, 0]):
    assert len(Ts) == len(Ks)

    camera_frames = o3d.geometry.LineSet()
    for T, K in zip(Ts, Ks):
        camera_frame = get_camera_ray_frame(T, K, size=size, color=color)
        camera_frames += camera_frame

    if len(Ts) > 1 and center_line:
        # TODO: fix open3d linese += when one of the line set is empty, the
        # color will be missing,
        center_line = get_camera_center_line(Ts, color=center_line_color)
        camera_frames += center_line

    return camera_frames

def create_sphere_lineset(radius=1.0, resolution=10, color=[0, 0, 0]):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius,
                                                          resolution=resolution)
    triangles = np.asarray(sphere_mesh.triangles)
    sphere_lineset = o3d.geometry.LineSet()
    sphere_lineset.points = sphere_mesh.vertices
    sphere_lineset.lines = o3d.utility.Vector2iVector(
        np.vstack((
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )))
    colors = np.empty((len(sphere_lineset.lines), 3))
    colors[:] = np.array(color)
    sphere_lineset.colors = o3d.utility.Vector3dVector(colors)
    return sphere_lineset
