import numpy as np
import open3d as o3d
from jaxtyping import Float, Int

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


def mesh_to_im_distance(
    mesh: o3d.geometry.TriangleMesh,
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
    height: int,
    width: int,
):
    """
    Cast mesh to a distance image given camera parameters and image dimensions.
    Each pixel contains the distance between camera center to the mesh surface.
    Invalid distances are set to np.inf.

    Args:
        mesh: Open3D mesh.
        K: (3, 3) array, camera intrinsic matrix.
        T: (4, 4) array, camera extrinsic matrix.
        height: int, image height.
        width: int, image width.

    Return:
        (height, width) array, float32, representing the distance image. The
        distance is the distance between camera center to the mesh surface.
        Invalid distances are set to np.inf.

    Note: to cast the same mesh to multiple set of camera parameters, use
    mesh_to_im_distances for higher efficiency.
    """
    im_distances = mesh_to_im_distances(
        mesh=mesh,
        Ks=[K],
        Ts=[T],
        height=height,
        width=width,
    )
    im_distance = im_distances[0]

    return im_distance


def mesh_to_im_distances(
    mesh: o3d.geometry.TriangleMesh,
    Ks: Float[np.ndarray, "n 3 3"],
    Ts: Float[np.ndarray, "n 4 4"],
    height: int,
    width: int,
):
    """
    Cast mesh to distance images given multiple camera parameters and image
    dimensions. Each distance image contains the distance between camera center
    to the mesh surface. Invalid distances are set to np.inf.

    Args:
        mesh: Open3D mesh.
        Ks: (N, 3, 3) array, camera intrinsic matrices.
        Ts: (N, 4, 4) array, camera extrinsic matrices.
        height: int, image height.
        width: int, image width.

    Return:
        (N, height, width) array, float32, representing the distance images. The
        distance is the distance between camera center to the mesh surface.
        Invalid distances are set to np.inf.
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

    im_distances = []
    for K, T in zip(Ks, Ts):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=K,
            extrinsic_matrix=T,
            width_px=width,
            height_px=height,
        )
        ray_lengths = np.linalg.norm(rays[:, :, 3:].numpy(), axis=2)
        ans = scene.cast_rays(rays)
        im_distance = ans["t_hit"].numpy() * ray_lengths
        im_distances.append(im_distance)
    im_distances = np.stack(im_distances, axis=0)

    return im_distances


def mesh_to_im_depth(
    mesh: o3d.geometry.TriangleMesh,
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
    height: int,
    width: int,
) -> Float[np.ndarray, "h w"]:
    """
    Cast mesh to a depth image given camera parameters and image dimensions.
    Each pixel contains the depth (z-coordinate) of the mesh surface in the
    camera frame. Invalid depths are set to np.inf.

    Args:
        mesh: Open3D mesh.
        K: (3, 3) array, camera intrinsic matrix.
        T: (4, 4) array, camera extrinsic matrix.
        height: int, image height.
        width: int, image width.

    Return:
        (height, width) array, float32, representing the depth image.
        Invalid depths are set to np.inf.
    """
    im_distance = mesh_to_im_distance(mesh, K, T, height, width)
    im_depth = convert.im_distance_to_im_depth(im_distance, K)
    return im_depth


def mesh_to_im_depths(
    mesh: o3d.geometry.TriangleMesh,
    Ks: Float[np.ndarray, "n 3 3"],
    Ts: Float[np.ndarray, "n 4 4"],
    height: int,
    width: int,
) -> Float[np.ndarray, "n h w"]:
    """
    Cast mesh to depth images given multiple camera parameters and image
    dimensions. Each depth image contains the depth (z-coordinate) of the mesh
    surface in the camera frame. Invalid depths are set to np.inf.

    Args:
        mesh: Open3D mesh.
        Ks: (N, 3, 3) array, camera intrinsic matrices.
        Ts: (N, 4, 4) array, camera extrinsic matrices.
        height: int, image height.
        width: int, image width.

    Return:
        (N, height, width) array, float32, representing the depth images.
        Invalid depths are set to np.inf.
    """
    im_distances = mesh_to_im_distances(mesh, Ks, Ts, height, width)
    im_depths = np.stack(
        [
            convert.im_distance_to_im_depth(im_distance, K)
            for im_distance, K in zip(im_distances, Ks)
        ],
        axis=0,
    )
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
    im_depth = mesh_to_im_distance(mesh, K, T, height, width)
    im_mask = (im_depth != np.inf).astype(np.float32)

    return im_mask
