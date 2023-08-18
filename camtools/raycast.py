import numpy as np
import open3d as o3d

from . import sanity
from . import convert


def gen_rays(K, T, pixels):
    """
    Args:
        pixels: image coordinates, (N, 2), order (col, row).

    Return:
        (centers, dirs)
        centers: camera center, (N, 3).
        dirs: ray directions, (N, 3), normalized to unit length.
    """
    sanity.assert_K(K)
    sanity.assert_T(T)
    sanity.assert_shape_nx2(pixels, name="pixels")

    # Concat xs_ys into homogeneous coordinates.
    points = np.concatenate([pixels, np.ones_like(pixels[:, :1])], axis=1)

    # Transform to camera space
    points = (np.linalg.inv(K) @ points.T).T

    # Normalize to have 1 distance
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    # Transform to world space
    R, _ = convert.T_to_R_t(T)
    C = convert.T_to_C(T)
    dirs = (np.linalg.inv(R) @ points.T).T

    # Tile camera center C
    centers = np.tile(C, (dirs.shape[0], 1))

    return centers, dirs


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
        (height, width) array, float32, representing depth image. Invalid depths
        are set to np.inf.

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
    ray_lengths = np.linalg.norm(rays[:, :, 3:].numpy(), axis=2)

    ans = scene.cast_rays(rays)
    im_depth = ans["t_hit"].numpy() * ray_lengths

    return im_depth


def mesh_to_depths(mesh, Ks, Ts, height, width):
    """
    Cast mesh to depth image given camera parameters and image dimensions.
    Same as mesh_to_depth, but for multiple cameras.

    Args:
        mesh: Open3D mesh.
        Ks: (N, 3, 3) array, camera intrinsic matrices.
        Ts: (N, 4, 4) array, camera extrinsic matrices.
        height: int, image height.
        width: int, image width.

    Return:
        (N, height, width) array, float32, representing depth image. Invalid
        depths are set to np.inf.
    """
    for K in Ks:
        sanity.assert_K(K)
    for T in Ts:
        sanity.assert_T(T)

    t_mesh = o3d.t.geometry.TriangleMesh(
        vertex_positions=np.asarray(mesh.vertices).astype(np.float32),
        triangle_indices=np.asarray(mesh.triangles),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    im_depths = []
    for K, T in zip(Ks, Ts):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=K,
            extrinsic_matrix=T,
            width_px=width,
            height_px=height,
        )
        ray_lengths = np.linalg.norm(rays[:, :, 3:].numpy(), axis=2)
        ans = scene.cast_rays(rays)
        im_depth = ans["t_hit"].numpy() * ray_lengths
        im_depths.append(im_depth)
    im_depths = np.stack(im_depths, axis=0)

    return im_depths


def mesh_to_mask(mesh, K, T, height, width):
    """
    Cast mesh to mask image given camera parameters and image dimensions.

    Args:
        mesh: Open3D mesh.
        K: (3, 3) array, camera intrinsic matrix.
        T: (4, 4) array, camera extrinsic matrix, [R | t] with [0, 0, 0, 1].
        height: int, image height.
        width: int, image width.

    Return:
        (height, width) array, float32, representing depth image. Foreground
        is set to 1.0. Background is set to 0.0.

    Note: this is not meant to be used repeatedly with the same mesh. If you
    need to perform ray casting of the same mesh multiple times, you should
    create the scene object manually to perform ray casting.
    """
    im_depth = mesh_to_depth(mesh, K, T, height, width)
    im_mask = (im_depth != np.inf).astype(np.float32)

    return im_mask
