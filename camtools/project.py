"""
Functions for projecting 2D->3D or 3D->2D.
"""

import cv2
import numpy as np
from jaxtyping import Float
from typing import Union, Tuple, Optional

from . import convert, image, sanity


def points_to_pixels(
    points: Float[np.ndarray, "n 3"],
    K: Float[np.ndarray, "3 3"],
    T: Float[np.ndarray, "4 4"],
) -> Float[np.ndarray, "N 2"]:
    """
    Project points in world coordinates to pixel coordinates.

    Example usage:
        pixels = ct.project.point_cloud_to_pixel(points, K, T)

        cols = pixels[:, 0]  # cols, width, x, top-left to top-right
        rows = pixels[:, 1]  # rows, height, y, top-left to bottom-left
        cols = np.round(cols).astype(np.int32)
        rows = np.round(rows).astype(np.int32)
        cols[cols >= width] = width - 1
        cols[cols < 0] = 0
        rows[rows >= height] = height - 1
        rows[rows < 0] = 0

    Args:
        K: (3, 3) array, camera intrinsic matrix.
        T: (4, 4) array, camera extrinsic matrix, [R | t] with [0, 0, 0, 1]
           below.
        points: (N, 3) array, 3D points in world coordinates.

    Return:
        (N, 2) array, representing [cols, rows] by each column. N is the number
        of points, which is not related to the image height and width.
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
    Convert a depth image to a point cloud, optionally including color information.
    Can return either a sparse (N, 3) point cloud or a dense one with the image
    shape (H, W, 3).

    Args:
        im_depth: Depth image (H, W), float32 or float64, in world scale.
        K: Intrinsics matrix (3, 3).
        T: Extrinsics matrix (4, 4).
        im_color: Color image (H, W, 3), float32/float64, range [0, 1].
        to_image: If True, returns a dense point cloud with the same shape as the
            input depth image (H, W, 3), while ignore_invalid is ignored as the
            invalid depths are not removed. If False, returns a sparse point cloud
            of shape (N, 3) while respecting ignore_invalid flag.
        ignore_invalid: If True, ignores invalid depths (<= 0 or >= inf).
        scale_factor: scale the im_depth (and optionally im_color) images before
            projecting to 3D points. When scale_factor == 0.5, the image size
            is reduced to half.

    Returns:
        - im_color == None, to_image == False:
            - return: points (N, 3)
        - im_color == None, to_image == True:
            - return: im_points (H, W, 3)
        - im_color != None, to_image == False:
            - return: (points (N, 3), colors (N, 3))
        - im_color != None, to_image == True:
            - return: (im_points (H, W, 3), im_colors (H, W, 3))
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
