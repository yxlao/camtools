import open3d as o3d
import numpy as np


def create_sphere_lineset(radius=1.0, resolution=10, color=[0, 0, 0]):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=resolution
    )
    triangles = np.asarray(sphere_mesh.triangles)
    sphere_lineset = o3d.geometry.LineSet()
    sphere_lineset.points = sphere_mesh.vertices
    sphere_lineset.lines = o3d.utility.Vector2iVector(
        np.vstack(
            (
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            )
        )
    )
    colors = np.empty((len(sphere_lineset.lines), 3))
    colors[:] = np.array(color)
    sphere_lineset.colors = o3d.utility.Vector3dVector(colors)
    return sphere_lineset


def mesh_to_lineset(mesh, downsample_ratio=1.0):
    # Downsample mesh
    if downsample_ratio < 1.0:
        target_number_of_triangles = int(len(mesh.triangles) * downsample_ratio)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles)
    elif downsample_ratio > 1.0:
        raise ValueError("Subsample must be less than or equal to 1.0")

    triangles = np.asarray(mesh.triangles)
    lineset = o3d.geometry.LineSet()
    lineset.points = mesh.vertices
    lineset.lines = o3d.utility.Vector2iVector(
        np.vstack(
            (
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            )
        )
    )
    colors = np.empty((len(lineset.lines), 3))
    colors[:] = np.array([0, 0, 0])
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset
