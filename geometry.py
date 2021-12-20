import open3d as o3d
import numpy as np
from . import convert


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


def get_camera_frames(Ts,
                      size=0.1,
                      color=[0, 0, 1],
                      center_line=True,
                      center_line_color=[1, 0, 0]):

    camera_frames = o3d.geometry.LineSet()
    for T in Ts:
        camera_frame = get_camera_frame(T, size=size, color=color)
        camera_frames += camera_frame

    if center_line:
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
