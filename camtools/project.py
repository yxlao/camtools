"""
Functions for projecting 2D->3D or 3D->2D.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
from jaxtyping import Float
from . import convert, image, sanity


def points_to_pixels(
    points: Float[np.ndarray, "n 3"],
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
) -> Float[np.ndarray, "n 2"]:
    """
    Project 3D points in world coordinates to 2D pixel coordinates using the
    camera intrinsic and extrinsic parameters.

    The projection follows the standard pinhole camera model:
        λ[x, y, 1]^T = K @ [R | t] @ [X, Y, Z, 1]^T
    where:
        - [X, Y, Z, 1]^T is a homogeneous 3D point in world coordinates
        - [R | t] is the 3x4 extrinsic matrix (world-to-camera transformation)
        - K is the 3x3 intrinsic matrix
        - [x, y, 1]^T is the projected homogeneous 2D point in pixel coordinates
        - λ is the depth value

    Example usage:
        pixels = ct.project.points_to_pixels(points, K, T)

        # Extract and round pixel coordinates
        cols = pixels[:, 0]  # x-coordinates (width dimension)
        rows = pixels[:, 1]  # y-coordinates (height dimension)
        cols = np.round(cols).astype(np.int32)
        rows = np.round(rows).astype(np.int32)

        # Clamp to image boundaries
        cols[cols >= width] = width - 1
        cols[cols < 0] = 0
        rows[rows >= height] = height - 1
        rows[rows < 0] = 0

    Args:
        points: (N, 3) array of 3D points in world coordinates.
        K: (3, 3) camera intrinsic matrix.
        T: (4, 4) camera extrinsic matrix (world-to-camera transformation).

    Returns:
        (N, 2) array of pixel coordinates, where each row contains [x, y]
        coordinates. The x-coordinate corresponds to the image width (columns)
        and the y-coordinate corresponds to the image height (rows).
    """
    sanity.assert_K(K)
    sanity.assert_T(T)
    sanity.assert_shape_nx3(points, name="points")

    W2P = convert.K_T_to_W2P(K, T)

    # (N, 3) -> (N, 4)
    points = convert.to_homo(points)
    # (N, 4)
    pixels = (W2P @ points.T).T
    # (N, 4) -> (N, 3), discard the last column
    pixels = pixels[:, :3]
    # (N, 3) -> (N, 2)
    pixels = convert.from_homo(pixels)

    return pixels


def im_depth_to_point_cloud(
    im_depth: Float[np.ndarray, "h w"],
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
    im_color: Optional[Float[np.ndarray, "h w 3"]] = None,
    to_image: bool = False,
    ignore_invalid: bool = True,
    scale_factor: float = 1.0,
) -> Union[
    Float[np.ndarray, "n 3"],
    Float[np.ndarray, "h w 3"],
    Tuple[Float[np.ndarray, "n 3"], Float[np.ndarray, "n 3"]],
    Tuple[Float[np.ndarray, "h w 3"], Float[np.ndarray, "h w 3"]],
]:
    """
    Convert a depth image to a 3D point cloud in world coordinates, optionally
    including color information. The point cloud can be returned in either a
    sparse format (N, 3) or a dense format matching the input image dimensions
    (H, W, 3).

    The conversion follows the inverse projection formula:
        [X, Y, Z]^T = pose @ (depth * inv(K) @ [u, v, 1]^T)
    where:
        - [u, v] are pixel coordinates
        - K is the intrinsic matrix
        - pose is the camera-to-world transformation matrix
        - depth is the depth value at pixel (u, v)

    Args:
        im_depth: (H, W) depth image in world scale, float32 or float64.
        K: (3, 3) camera intrinsic matrix.
        T: (4, 4) camera extrinsic matrix (world-to-camera transformation).
        im_color: Optional (H, W, 3) color image in range [0, 1], float32/float64.
        to_image: If True, returns a dense point cloud with shape (H, W, 3).
            If False, returns a sparse point cloud with shape (N, 3).
        ignore_invalid: If True, filters out points with invalid depths
            (<= 0 or >= inf).
        scale_factor: Scaling factor for the input images. When scale_factor < 1,
            the images are downsampled and the intrinsic matrix is adjusted
            accordingly.

    Returns:
        Depending on the input parameters, returns one of:
        - im_color == None, to_image == False:
            points: (N, 3) array of 3D points
        - im_color == None, to_image == True:
            im_points: (H, W, 3) array of 3D points
        - im_color != None, to_image == False:
            (points, colors): Tuple of (N, 3) arrays for points and colors
        - im_color != None, to_image == True:
            (im_points, im_colors): Tuple of (H, W, 3) arrays for points and colors
    """
    # Sanity checks
    sanity.assert_K(K)
    sanity.assert_T(T)
    if not isinstance(im_depth, np.ndarray):
        raise TypeError("im_depth must be a numpy array")
    if im_depth.dtype not in [np.float32, np.float64]:
        raise TypeError("im_depth must be of type float32 or float64")
    if im_depth.ndim != 2:
        raise ValueError("im_depth must be a 2D array")
    if im_color is not None:
        if not isinstance(im_color, np.ndarray):
            raise TypeError("im_color must be a numpy array")
        if im_color.shape[:2] != im_depth.shape or im_color.ndim != 3:
            raise ValueError(
                f"im_color must be (H, W, 3), and have the same "
                f"shape as im_depth, but got {im_color.shape}."
            )
        if im_color.dtype not in [np.float32, np.float64]:
            raise TypeError("im_color must be of type float32 or float64")
        if im_color.max() > 1.0 or im_color.min() < 0.0:
            raise ValueError("im_color values must be in the range [0, 1]")
    if to_image and ignore_invalid:
        print("Warning: ignore_invalid is ignored when to_image is True.")
        ignore_invalid = False

    # Make copies as K may be modified inplace
    K = np.copy(K)
    T = np.copy(T)

    if scale_factor != 1.0:
        # Calculate new dimensions
        new_width = int(im_depth.shape[1] * scale_factor)
        new_height = int(im_depth.shape[0] * scale_factor)

        # Resize images
        im_depth = image.resize(
            im_depth,
            shape_wh=(new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )
        if im_color is not None:
            im_color = image.resize(
                im_color,
                shape_wh=(new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # Adjust the intrinsic matrix K for the new image dimensions
        K[0, 0] *= scale_factor
        K[1, 1] *= scale_factor
        K[0, 2] *= scale_factor
        K[1, 2] *= scale_factor

    height, width = im_depth.shape
    pose = convert.T_to_pose(T)

    # pixels.shape == (height, width, 2)
    # pixels[r, c] == [c, r], since x-axis goes from top-left to top-right.
    pixels = np.transpose(np.indices((width, height)), (2, 1, 0))
    # (height * width, 2)
    pixels = pixels.reshape((-1, 2))
    # (height * width, 3)
    pixels_homo = convert.to_homo(pixels)
    # (height * width, )
    depths = im_depth.flatten()

    if ignore_invalid:
        valid_mask = (depths > 0) & (depths < np.inf)
        depths = depths[valid_mask]
        pixels_homo = pixels_homo[valid_mask]
        if im_color is not None:
            colors = im_color.reshape((-1, 3))[valid_mask]

    # Transform pixel coordinates to world coordinates.
    # (height * width, 1)
    depths = depths.reshape((-1, 1))

    # (N, 3)
    points_camera = depths * (np.linalg.inv(K) @ pixels_homo.T).T
    # (N, 4)
    points_world = (pose @ (convert.to_homo(points_camera).T)).T
    # (N, 3)
    points_world = convert.from_homo(points_world)

    if to_image:
        assert (
            ignore_invalid == False
        ), "ignore_invalid is ignored when to_image is True."
        points_world = points_world.reshape((height, width, 3))
        if im_color is None:
            return points_world
        else:
            return points_world, im_color
    else:
        if im_color is None:
            return points_world
        else:
            return points_world, colors
