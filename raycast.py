import numpy as np
import open3d as o3d

from . import sanity


def mesh_to_depth(mesh, K, T, height, width):
    """
    Cast mesh to depth image given camera parameters and image dimensions.

    Args:
        mesh: Open3D mesh.
        K: (3, 3) array, camera intrinsic matrix.
        T: (4, 4) array, camera extrinsic matrix, [R | t] with [0, 0, 0, 1].
        height: int, image height.
        width: int, image width.

    Return:
        (height, width) array, float32, representing depth image. Invalid depth
        is set to np.inf.

    Note: this is not meant to be used repeatedly with the same mesh. If you
    need to perform ray casting of the same mesh multiple times, you should
    create the scene object manually to perform ray casting.
    """
    sanity.assert_K(K)
    sanity.assert_T(T)

    t_mesh = o3d.t.geometry.TriangleMesh(
        vertex_positions=np.asarray(mesh.vertices).astype(np.float32),
        triangle_indices=np.asarray(mesh.triangles),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=K,
        extrinsic_matrix=T,
        width_px=width,
        height_px=height,
    )
    ans = scene.cast_rays(rays)
    im_depth = ans['t_hit'].numpy()

    return im_depth
