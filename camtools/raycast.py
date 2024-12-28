import numpy as np
import open3d as o3d
from typing import Tuple
from jaxtyping import Float

from . import sanity
from . import convert


def gen_rays(
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
    pixels: Float[np.ndarray, "n 2"],
) -> Tuple[Float[np.ndarray, "n 3"], Float[np.ndarray, "n 3"]]:
    """
    Generate camera rays in world coordinates for given pixel coordinates.

    The rays are generated using the pinhole camera model:
        [X, Y, Z]^T = pose @ (inv(K) @ [u, v, 1]^T)
    where:
        - [u, v] are pixel coordinates
        - K is the intrinsic matrix
        - pose is the camera-to-world transformation matrix
        - [X, Y, Z] is the ray direction in world coordinates

    Example usage:
        # Generate rays for all pixels in a 640x480 image
        height, width = 480, 640
        pixels = np.array([[x, y] for y in range(height) for x in range(width)])
        centers, dirs = ct.raycast.gen_rays(K, T, pixels)

    Args:
        K: (3, 3) camera intrinsic matrix.
        T: (4, 4) camera extrinsic matrix (world-to-camera transformation).
        pixels: (N, 2) array of pixel coordinates in (col, row) order.

    Returns:
        Tuple containing:
        - centers: (N, 3) array of camera centers in world coordinates. All
          centers are identical since they originate from the same camera.
        - dirs: (N, 3) array of ray directions in world coordinates, normalized
          to unit length.
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
) -> Float[np.ndarray, "h w"]:
    """
    Generate a distance image by ray casting a mesh from a given camera view.

    The distance image contains the Euclidean distance from the camera center to
    the mesh surface for each pixel. The ray casting follows the equation:
        distance = ||C - P||
    where:
        - C is the camera center in world coordinates
        - P is the intersection point on the mesh surface
        - ||·|| denotes the Euclidean norm

    Args:
        mesh (o3d.geometry.TriangleMesh): Open3D TriangleMesh to be ray casted.
        K (Float[np.ndarray, "3 3"]): Camera intrinsic matrix.
        T (Float[np.ndarray, "4 4"]): Camera extrinsic matrix (world-to-camera
            transformation).
        height (int): Image height in pixels.
        width (int): Image width in pixels.

    Returns:
        Float[np.ndarray, "h w"]: Distance image as a float32 array with shape
            (height, width). Each pixel contains the distance from the camera
            center to the mesh surface. Invalid distances (no intersection) are
            set to np.inf.

    Example:
        >>> # Create distance image for a 640x480 view
        >>> distance_image = ct.raycast.mesh_to_im_distance(mesh, K, T, 480, 640)
        >>> # Visualize distances
        >>> plt.imshow(distance_image)
        >>> plt.colorbar()

    Note:
        For casting the same mesh with multiple camera views, use
        mesh_to_im_distances for better efficiency as it avoids repeated scene
        setup.
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
) -> Float[np.ndarray, "n h w"]:
    """
    Generate multiple distance images by ray casting a mesh from different views.

    For each camera view, generates a distance image containing the Euclidean
    distance from the camera center to the mesh surface. The distances are
    calculated as:
        distance = ||C_i - P_i||
    where:
        - C_i is the camera center for view i
        - P_i is the intersection point on the mesh surface for view i
        - ||·|| denotes the Euclidean norm

    Args:
        mesh (o3d.geometry.TriangleMesh): Open3D TriangleMesh to be ray casted.
        Ks (Float[np.ndarray, "n 3 3"]): Array of camera intrinsic matrices for
            N views.
        Ts (Float[np.ndarray, "n 4 4"]): Array of camera extrinsic matrices
            (world-to-camera transformations) for N views.
        height (int): Image height in pixels.
        width (int): Image width in pixels.

    Returns:
        Float[np.ndarray, "n h w"]: Distance images as a float32 array with shape
            (N, height, width). Each image contains the distances from the
            corresponding camera center to the mesh surface. Invalid distances
            (no intersection) are set to np.inf.

    Example:
        >>> # Create distance images for 3 different views
        >>> distances = ct.raycast.mesh_to_im_distances(mesh, Ks, Ts, 480, 640)
        >>> # Visualize first view's distances
        >>> plt.imshow(distances[0])
        >>> plt.colorbar()

    Note:
        This function is more efficient than calling mesh_to_im_distance multiple
        times as it only sets up the ray casting scene once.
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
    Generate a depth image by ray casting a mesh from a given camera view.

    The depth image contains the z-coordinate of the mesh surface in the camera
    coordinate system for each pixel. The depth is calculated as:
        depth = (distance * f) / sqrt(u² + v² + f²)
    where:
        - distance is the Euclidean distance from camera center to surface point
        - f is the focal length from the intrinsic matrix K
        - (u, v) are the pixel coordinates in the camera plane

    Args:
        mesh (o3d.geometry.TriangleMesh): Open3D TriangleMesh to be ray casted.
        K (Float[np.ndarray, "3 3"]): Camera intrinsic matrix.
        T (Float[np.ndarray, "4 4"]): Camera extrinsic matrix (world-to-camera
            transformation).
        height (int): Image height in pixels.
        width (int): Image width in pixels.

    Returns:
        Float[np.ndarray, "h w"]: Depth image as a float32 array with shape
            (height, width). Each pixel contains the z-coordinate of the mesh
            surface in camera space. Invalid depths (no intersection) are set
            to np.inf.

    Example:
        >>> # Create depth image for a 640x480 view
        >>> depth_image = ct.raycast.mesh_to_im_depth(mesh, K, T, 480, 640)
        >>> # Visualize depths
        >>> plt.imshow(depth_image)
        >>> plt.colorbar()

    Note:
        This function internally uses mesh_to_im_distance and converts the
        distances to depths using the camera intrinsic parameters.
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
    Generate multiple depth images by ray casting a mesh from different views.

    For each camera view, generates a depth image containing the z-coordinate of
    the mesh surface in the camera coordinate system. The depths are calculated as:
        depth = (distance * f) / sqrt(u² + v² + f²)
    where:
        - distance is the Euclidean distance from camera center to surface point
        - f is the focal length from the intrinsic matrix K
        - (u, v) are the pixel coordinates in the camera plane

    Args:
        mesh (o3d.geometry.TriangleMesh): Open3D TriangleMesh to be ray casted.
        Ks (Float[np.ndarray, "n 3 3"]): Array of camera intrinsic matrices for
            N views.
        Ts (Float[np.ndarray, "n 4 4"]): Array of camera extrinsic matrices
            (world-to-camera transformations) for N views.
        height (int): Image height in pixels.
        width (int): Image width in pixels.

    Returns:
        Float[np.ndarray, "n h w"]: Depth images as a float32 array with shape
            (N, height, width). Each image contains the z-coordinates of the mesh
            surface in the corresponding camera space. Invalid depths (no
            intersection) are set to np.inf.

    Example:
        >>> # Create depth images for 3 different views
        >>> depths = ct.raycast.mesh_to_im_depths(mesh, Ks, Ts, 480, 640)
        >>> # Visualize first view's depths
        >>> plt.imshow(depths[0])
        >>> plt.colorbar()

    Note:
        This function internally uses mesh_to_im_distances and converts the
        distances to depths using the camera intrinsic parameters.
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


def mesh_to_im_mask(
    mesh: o3d.geometry.TriangleMesh,
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
    height: int,
    width: int,
) -> Float[np.ndarray, "h w"]:
    """
    Generate a binary mask image by ray casting a mesh from a given camera view.

    The mask image indicates which pixels contain the mesh (foreground) and which
    do not (background). Foreground pixels are set to 1.0 and background pixels
    are set to 0.0.

    Example usage:
        # Create mask image for a 640x480 view
        mask = ct.raycast.mesh_to_im_mask(mesh, K, T, 480, 640)
        # Visualize mask
        plt.imshow(mask, cmap='gray')

    Args:
        mesh: Open3D TriangleMesh to be ray casted.
        K: (3, 3) camera intrinsic matrix.
        T: (4, 4) camera extrinsic matrix (world-to-camera transformation).
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        (height, width) float32 array representing the binary mask image.
        Foreground pixels (mesh visible) are set to 1.0, background pixels
        (no mesh) are set to 0.0.

    Note: This function is not optimized for repeated use with the same mesh.
    For multiple ray casts with the same mesh, create the ray casting scene
    manually for better performance.
    """
    im_distance = mesh_to_im_distance(mesh, K, T, height, width)
    im_mask = (im_distance != np.inf).astype(np.float32)

    return im_mask
