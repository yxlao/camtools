import open3d as o3d
import numpy as np
from jaxtyping import Float
from typing import Optional


def create_sphere_lineset(
    radius: float = 1.0,
    resolution: int = 10,
    color: Float[np.ndarray, "3"] = np.array([0, 0, 0]),
) -> o3d.geometry.LineSet:
    """
    Create a sphere represented as a line set.

    Args:
        radius: Radius of the sphere.
        resolution: Resolution of the sphere mesh.
        color: RGB color of the sphere lines.

    Returns:
        Open3D LineSet representing the sphere.
    """
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


def mesh_to_lineset(
    mesh: o3d.geometry.TriangleMesh,
    downsample_ratio: float = 1.0,
    color: Optional[Float[np.ndarray, "3"]] = None,
) -> o3d.geometry.LineSet:
    """
    Convert a mesh to a line set, optionally downsampling it.

    Args:
        mesh: Open3D triangle mesh to convert.
        downsample_ratio: Ratio of triangles to keep (0.0 to 1.0).
        color: Optional RGB color for the lines. If None, uses black.

    Returns:
        Open3D LineSet representing the mesh edges.

    Raises:
        ValueError: If downsample_ratio is greater than 1.0.
    """
    # Downsample mesh
    if downsample_ratio < 1.0:
        target_number_of_triangles = int(
            len(mesh.triangles) * downsample_ratio
        )
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
    colors[:] = np.array(color) if color is not None else np.array([0, 0, 0])
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset
