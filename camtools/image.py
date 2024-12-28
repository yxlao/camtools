import numpy as np
import cv2
from . import sanity
from . import colormap
from typing import Tuple, List, Optional, Union
from jaxtyping import Float, UInt8, UInt16, Int


def crop_white_boarders(
    im: Float[np.ndarray, "h w 3"], padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> Float[np.ndarray, "h_cropped w_cropped 3"]:
    """
    Crop white boarders from an image.

    Args:
        im: Image, float32.
        padding: Padding to apply after cropping.

    Return:
        Cropped and padded image.
    """
    tblr = compute_cropping(im)
    im_dst = apply_cropping_padding(im, tblr, padding)
    return im_dst


def compute_cropping_v1(im: Float[np.ndarray, "h w n"]) -> Tuple[int, int, int, int]:
    """
    Compute top, bottom, left, right white boarder in pixels.

    This function can handle (H, W, N) images, e.g.,
    - 3-channel image: (H, W, 3)
    - 3-channel images concatenated in the 2nd dimension: (H, W, 3 x num_im)

    Args:
        im: Image, float32.

    Return: tuple of 4 elements
        crop_t: int, number of white pixels on the top edge.
        crop_b: int, number of white pixels on the bottom edge.
        crop_l: int, number of white pixels on the left edge.
        crop_r: int, number of white pixels on the right edge.
    """
    if not im.dtype == np.float32:
        raise ValueError(f"im.dtype == {im.dtype} != np.float32")
    if not im.ndim == 3:
        raise ValueError(f"im must be (H, W, N), but got {im.shape}")
    if im.shape[2] == 0:
        raise ValueError(f"Empty image, got {im.shape}")

    h, w, _ = im.shape

    # Find the number of white pixels on each edge.
    crop_t = 0
    crop_b = 0
    crop_l = 0
    crop_r = 0

    for t in range(h):
        if np.allclose(im[t, :, :], 1.0):
            crop_t += 1
        else:
            break
    for b in range(h):
        if np.allclose(im[h - b - 1, :, :], 1.0):
            crop_b += 1
        else:
            break
    for l in range(w):
        if np.allclose(im[:, l, :], 1.0):
            crop_l += 1
        else:
            break
    for r in range(w):
        if np.allclose(im[:, w - r - 1, :], 1.0):
            crop_r += 1
        else:
            break

    return crop_t, crop_b, crop_l, crop_r


def compute_cropping(
    im: Float[np.ndarray, "h w 3"], check_with_v1: bool = False
) -> Tuple[int, int, int, int]:
    """
    Compute top, bottom, left, right white borders in pixels for a 3-channel
    image.

    This function is designed for (H, W, 3) images, where each pixel's value
    ranges from 0.0 to 1.0, and white pixels are represented by (1.0, 1.0, 1.0).

    Args:
        im: Input image as a NumPy array of dtype float32.
        check_with_v1: If True, verify results against compute_cropping_v1.

    Returns:
        A tuple containing the number of white pixels to crop from the top,
        bottom, left, and right edges, respectively.
    """
    if not im.dtype == np.float32:
        raise ValueError(f"Expected im.dtype to be np.float32, but got {im.dtype}")
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError(f"Expected im to be of shape (H, W, 3), but got {im.shape}")

    # Create a mask where white pixels are marked as True
    white_mask = np.all(im == 1.0, axis=-1)

    # Find the indices of rows and columns where there's at least one non-white pixel
    rows_with_color = np.where(~white_mask.all(axis=1))[0]
    cols_with_color = np.where(~white_mask.all(axis=0))[0]

    # Determine the crop values based on the positions of non-white pixels
    crop_t = rows_with_color[0] if len(rows_with_color) else 0
    crop_b = im.shape[0] - rows_with_color[-1] - 1 if len(rows_with_color) else 0
    crop_l = cols_with_color[0] if len(cols_with_color) else 0
    crop_r = im.shape[1] - cols_with_color[-1] - 1 if len(cols_with_color) else 0

    # Check the results against compute_cropping_v1 if requested
    if check_with_v1:
        crop_t_v1, crop_b_v1, crop_l_v1, crop_r_v1 = compute_cropping_v1(im)
        if (
            crop_t != crop_t_v1
            or crop_b != crop_b_v1
            or crop_l != crop_l_v1
            or crop_r != crop_r_v1
        ):
            raise ValueError(
                f"compute_cropping_v1 failed to compute the correct cropping: "
                f"({crop_t}, {crop_b}, {crop_l}, {crop_r}) != "
                f"({crop_t_v1}, {crop_b_v1}, {crop_l_v1}, {crop_r_v1})"
            )

    return crop_t, crop_b, crop_l, crop_r


def apply_cropping_padding(
    im_src: Float[np.ndarray, "h w 3"],
    cropping: Tuple[int, int, int, int],
    padding: Tuple[int, int, int, int],
) -> Float[np.ndarray, "h_cropped w_cropped 3"]:
    """
    Apply cropping and padding to an image.

    Args:
        im_src: Image, float32.
        cropping: 4-tuple (crop_t, crop_b, crop_l, crop_r)
        padding: 4-tuple (pad_t, pad_b, pad_l, pad_r)

    Return:
        Cropped and padded image.
    """
    if not im_src.dtype == np.float32:
        raise ValueError(f"im_src.dtype == {im_src.dtype} != np.float32")
    if not im_src.ndim == 3:
        raise ValueError(f"im_src must be (H, W, 3), but got {im_src.shape}")

    (
        h,
        w,
        _,
    ) = im_src.shape
    crop_t, crop_b, crop_l, crop_r = cropping
    im_dst = im_src[crop_t : h - crop_b, crop_l : w - crop_r, :]
    pad_t, pad_b, pad_l, pad_r = padding
    im_dst = np.pad(
        im_dst,
        ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
        mode="constant",
        constant_values=1.0,
    )
    return im_dst


def apply_croppings_paddings(
    src_ims: List[Float[np.ndarray, "h w 3"]],
    croppings: List[Tuple[int, int, int, int]],
    paddings: List[Tuple[int, int, int, int]],
) -> List[Float[np.ndarray, "h_cropped w_cropped 3"]]:
    """
    Apply cropping and padding to a list of images.

    Args:
        src_ims: list of images, float32.
        croppings: list of 4-tuples
            [
                (crop_t, crop_b, crop_l, crop_r),
                (crop_t, crop_b, crop_l, crop_r),
                ...
            ]
        paddings: list of 4-tuples
            [
                (pad_t, pad_b, pad_l, pad_r),
                (pad_t, pad_b, pad_l, pad_r),
                ...
            ]
    """
    num_ims = len(src_ims)
    if not len(croppings) == num_ims:
        raise ValueError(f"len(croppings) == {len(croppings)} != {num_ims}")
    if not len(paddings) == num_ims:
        raise ValueError(f"len(paddings) == {len(paddings)} != {num_ims}")
    for cropping in croppings:
        if not len(cropping) == 4:
            raise ValueError(f"len(cropping) == {len(cropping)} != 4")

    dst_ims = []
    for im_src, cropping, padding in zip(src_ims, croppings, paddings):
        im_dst = apply_cropping_padding(im_src, cropping, padding)
        dst_ims.append(im_dst)

    return dst_ims


def get_post_croppings_paddings_shapes(
    src_shapes: List[Tuple[int, int, int]],
    croppings: List[Tuple[int, int, int, int]],
    paddings: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int]]:
    """
    Compute image shapes after cropping and padding.

    Args:
        src_shapes: list of source image shapes.
        croppings: list of 4-tuples
            [
                (crop_t, crop_b, crop_l, crop_r),
                (crop_t, crop_b, crop_l, crop_r),
                ...
            ]
        paddings: list of 4-tuples
            [
                (pad_t, pad_b, pad_l, pad_r),
                (pad_t, pad_b, pad_l, pad_r),
                ...
            ]
    """
    dst_shapes = []
    for src_shape, cropping, padding in zip(src_shapes, croppings, paddings):
        crop_t, crop_b, crop_l, crop_r = cropping
        pad_t, pad_b, pad_l, pad_r = padding
        dst_shape = (
            src_shape[0] - crop_t - crop_b + pad_t + pad_b,
            src_shape[1] - crop_l - crop_r + pad_l + pad_r,
            src_shape[2],
        )
        dst_shapes.append(dst_shape)
    return dst_shapes


def overlay_mask_on_rgb(
    im_rgb: Float[np.ndarray, "h w 3"],
    im_mask: Float[np.ndarray, "h w"],
    overlay_alpha: float = 0.4,
    overlay_color: Float[np.ndarray, "3"] = np.array([0, 0, 1]),
) -> Float[np.ndarray, "h w 3"]:
    """
    Overlay mask on to of RGB image.

    Args:
        im_rgb: RGB image, 3 channels, float, range from 0 to 1.
        im_mask: Mask image, 1 channel, float, range from 0 to 1.
        overlay_alpha: Alpha value for overlay, float, range from 0 to 1.
        overlay_color: Color of overlay, 3 channels, float, range from 0 to 1.
    """
    # Sanity: im_rgb
    assert im_rgb.shape[:2] == im_mask.shape
    assert im_rgb.dtype == np.float32 or im_rgb.dtype == np.float64
    assert im_rgb.max() <= 1.0 and im_rgb.min() >= 0.0

    # Sanity: im_mask
    assert im_mask.dtype == np.float32 or im_mask.dtype == np.float64
    assert im_mask.max() <= 1.0 and im_mask.min() >= 0.0

    # Sanity: overlay_alpha
    assert overlay_alpha >= 0.0 and overlay_alpha <= 1.0

    # Sanity: overlay_color
    overlay_color = np.array(overlay_color)
    assert overlay_color.shape == (3,)
    assert overlay_color.max() <= 1.0 and overlay_color.min() >= 0.0

    im_mask_stacked = np.dstack([im_mask, im_mask, im_mask])
    im_hard = im_rgb * (1.0 - im_mask_stacked) + overlay_color * im_mask_stacked
    im_soft = im_rgb * (1.0 - overlay_alpha) + im_hard * overlay_alpha

    return im_soft


def ndc_coords_to_pixels(
    ndc_coords: Float[np.ndarray, "n 2"],
    im_size_wh: Tuple[int, int],
    align_corners: bool = False,
) -> Float[np.ndarray, "n 2"]:
    """
    Convert NDC coordinates (from -1 to 1) to pixel coordinates. Out-of-bound
    values will NOT be corrected.

    Args:
        ndc_coords: NDC coordinates. Each row represents (x, y) or (c, r).
            Most values shall be in [-1, 1], where (-1, -1) is the top left
            corner and (1, 1) is the bottom right corner.
        im_size_wh: Image size (width, height).
        align_corners: If True, -1 and 1 are aligned to the center of the corner
            pixels. If False, -1 and 1 are aligned to the corner of the corner
            pixels. In general image interpolation, if align_corners=True, the
            src and dst images are aligned by the center point of their corner
            pixels. If align_corners=False, the src and dst images are aligned
            by the corner points of the corner pixels. Here, the NDC space does
            not have a "pixels size", and thus we precisely align the extrema -1
            and 1 to the center or corner of the corner pixels.

    Returns:
        Pixel coordinates.
    """
    sanity.assert_shape(ndc_coords, (None, 2), name="ndc_coords")
    w, h = im_size_wh[:2]
    dtype = ndc_coords.dtype

    src_tl = np.array([-1.0, -1.0], dtype=dtype)
    src_br = np.array([1.0, 1.0], dtype=dtype)

    if align_corners:
        # (-1, -1) -> (    0,     0)
        # (1 ,  1) -> (w - 1, h - 1)
        dst_tl = np.array([0, 0], dtype=dtype)
        dst_br = np.array([w - 1, h - 1], dtype=dtype)
    else:
        # (-1, -1) -> (   -0.5,    -0.5)
        # (1 ,  1) -> (w - 0.5, h - 0.5)
        # Align to the corner of the corner pixels.
        dst_tl = np.array([-0.5, -0.5], dtype=dtype)
        dst_br = np.array([w - 0.5, h - 0.5], dtype=dtype)

    dst_pixels = (ndc_coords - src_tl) / (src_br - src_tl) * (dst_br - dst_tl) + dst_tl

    return dst_pixels


def rotate(
    im: Float[np.ndarray, "h w c"], ccw_degrees: int
) -> Float[np.ndarray, "h_rotated w_rotated c"]:
    """
    Rotate an image counter-clockwise by a given angle.

    Args:
        im: The image to rotate.
        ccw_degrees: Counter-clockwise rotation angle in degrees, must be
            0, 90, 180, or 270.

    Returns:
        Rotated image.
    """
    if ccw_degrees == 0:
        im_rotated = np.copy(im)
    elif ccw_degrees == 90:
        im_rotated = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif ccw_degrees == 180:
        im_rotated = cv2.rotate(im, cv2.ROTATE_180)
    elif ccw_degrees == 270:
        im_rotated = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError(f"Invalid rotation angle: {ccw_degrees}.")

    return im_rotated


def recover_rotated_pixels(dst_pixels, src_wh, ccw_degrees):
    """
    Converts dst image pixel coordinates to src image pixel coordinates, where
    the dst image is the result of rotating the src image counter-clockwise by
    a given ccw_degrees angle.

    Args:
        dst_pixels: Pixel coordinates in the dst image. (N, 2) array of floats.
            Each row is (c, r).
        src_wh: Width and height of the src image, 2-element tuple of ints.
        ccw_degrees: Counter-clockwise rotation angle in degrees, must be
            0, 90, 180, or 270.

    Returns:
        Pixel coordinates in the src image. (N, 2) array of floats.

    Notes:
        1. This function is paired with OpenCV's cv2.resize() function, where
           the *center* of the top-left pixel is considered to be (0, 0).
           - Top-left     corner: (-0.5   , -0.5   )
           - Bottom-right corner: (w - 0.5, h - 0.5)
           However, most other implementations in computer graphics treat the
           *corner* of the top-left pixel to be (0, 0). For more discussions, see:
           https://www.realtimerendering.com/blog/the-center-of-the-pixel-is-0-50-5/
        2. OpenCV's image size is (width, height), while numpy's array shape
           is (height, width) or (height, width, 3). Be careful with the order.
    """
    # - src:
    #   - src_wh    : (w    ,     h)
    #   - src_pixels: (c    ,     r)
    # - dst:
    #   - dst_pixels: (r    , w-1-c)  # rotate 90
    #   - dst_pixels: (w-1-c, h-1-r)  # rotate 180
    #   - dst_pixels: (h-1-r,     c)  # rotate 270
    sanity.assert_shape(dst_pixels, (None, 2), name="dst_pixels")
    w, h = src_wh

    # Convert back to src.
    dst_c = dst_pixels[:, 0]
    dst_r = dst_pixels[:, 1]
    if ccw_degrees == 0:
        src_pixels = np.copy(dst_pixels)
    elif ccw_degrees == 90:
        src_pixels = np.stack([w - 1 - dst_r, dst_c], axis=1)
    elif ccw_degrees == 180:
        src_pixels = np.stack([w - 1 - dst_c, h - 1 - dst_r], axis=1)
    elif ccw_degrees == 270:
        src_pixels = np.stack([dst_r, h - 1 - dst_c], axis=1)
    else:
        raise ValueError(f"Invalid rotation angle: {ccw_degrees}.")

    # Sanity check.
    src_c = src_pixels[:, 0]
    src_r = src_pixels[:, 1]
    if ccw_degrees == 0:
        dst_pixels_recovered = np.copy(src_pixels)
    elif ccw_degrees == 90:
        dst_pixels_recovered = np.stack([src_r, w - 1 - src_c], axis=1)
    elif ccw_degrees == 180:
        dst_pixels_recovered = np.stack([w - 1 - src_c, h - 1 - src_r], axis=1)
    elif ccw_degrees == 270:
        dst_pixels_recovered = np.stack([h - 1 - src_r, src_c], axis=1)
    else:
        raise ValueError(f"Invalid rotation angle: {ccw_degrees}.")
    np.testing.assert_allclose(dst_pixels, dst_pixels_recovered, rtol=1e-5, atol=1e-5)

    return src_pixels


def resize(
    im: Union[
        Float[np.ndarray, "h w"],
        Float[np.ndarray, "h w 3"],
        UInt8[np.ndarray, "h w"],
        UInt8[np.ndarray, "h w 3"],
        UInt16[np.ndarray, "h w"],
        UInt16[np.ndarray, "h w 3"],
    ],
    shape_wh: Tuple[int, int],
    aspect_ratio_fill: Optional[
        Union[float, Tuple[float, float, float], np.ndarray]
    ] = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> Union[
    Float[np.ndarray, "h_ w_"],
    Float[np.ndarray, "h_ w_ 3"],
    UInt8[np.ndarray, "h_ w_"],
    UInt8[np.ndarray, "h_ w_ 3"],
    UInt16[np.ndarray, "h_ w_"],
    UInt16[np.ndarray, "h_ w_ 3"],
]:
    """
    Resize image to shape_wh = (width, height).
    In numpy, the resulting shape is (height, width) or (height, width, 3).

    Args:
        im: Numpy shape {(h, w), (h, w, 3)};
            dtype {uint8, uint16, float32, float64}.
        shape_wh: Tuple of (width, height). Be careful with the order.
        aspect_ratio_fill: The value to fill in order to keep the aspect ratio.
            - If None, image will be directly resized to (height, width).
            - If not None, the number of elements must match the channel size,
              1 or 3. The dtype and range must also match the input image.
              These value(s) will be filled to maintain the aspect ratio.
        interpolation: OpenCV interpolation method, e.g., cv2.INTER_LINEAR.

    Returns:
        im_resized: image of shape (height, width) or (height, width, 3).

    Notes:
        OpenCV's image size is (width, height), while numpy's array shape is
        (height, width) or (height, width, 3). Be careful with the order.
    """
    # Sanity: dtype.
    dtype = im.dtype
    assert dtype in (np.uint8, np.uint16, np.float32, np.float64)

    # Sanity: input shape.
    ndim = im.ndim
    assert ndim in {2, 3}, "ndim must be 2 or 3"
    if ndim == 3:
        assert im.shape[2] == 3, "im.shape[2] must be 3"

    # Sanity: output shape.
    dst_w, dst_h = shape_wh
    assert dst_w > 0 and dst_h > 0
    if ndim == 2:
        dst_numpy_shape = (dst_h, dst_w)
    else:
        dst_numpy_shape = (dst_h, dst_w, 3)

    # Sanity: aspect_ratio_fill's shape and value.
    if aspect_ratio_fill is not None:
        aspect_ratio_fill = np.array(aspect_ratio_fill).flatten()
        if ndim == 2:
            assert len(aspect_ratio_fill) == 1
        else:
            assert len(aspect_ratio_fill) == 3
        if dtype == np.float32 or dtype == np.float64:
            assert aspect_ratio_fill.max() <= 1.0
            assert aspect_ratio_fill.min() >= 0.0
        aspect_ratio_fill = aspect_ratio_fill.astype(dtype)

    # Compute intermediate shape (tmp_w, tmp_h)
    if aspect_ratio_fill is None:
        # Case 1: direct reshape, do not keep aspect ratio.
        tmp_w, tmp_h = dst_w, dst_h
    else:
        # Case 2; keep aspect ratio and fill with aspect_ratio_fill.
        src_h, src_w = im.shape[:2]
        src_wh_ratio = src_w / float(src_h)
        dst_wh_ratio = dst_w / float(dst_h)
        if src_wh_ratio >= dst_wh_ratio:
            # Source image is "wider". Pad in the height dimension.
            tmp_w = dst_w
            tmp_h = int(round(tmp_w / src_wh_ratio))
        else:
            # Source image is "taller". Pad in the width dimension.
            tmp_h = dst_h
            tmp_w = int(round(tmp_h * src_wh_ratio))
        assert tmp_w <= dst_w and tmp_h <= dst_h

    # Resize.
    im_tmp = cv2.resize(im, dsize=(tmp_w, tmp_h), interpolation=interpolation)

    # Pad if necessary.
    if tmp_w == dst_w and tmp_h == dst_h:
        im_resize = im_tmp
    else:
        im_resize = np.full(dst_numpy_shape, fill_value=aspect_ratio_fill, dtype=dtype)
        im_resize[:tmp_h, :tmp_w] = im_tmp

    # Final sanity checks for the reshaped image.
    assert im_resize.shape == dst_numpy_shape

    return im_resize


def recover_resized_pixels(
    dst_pixels: Float[np.ndarray, "n 2"],
    src_wh: Tuple[int, int],
    dst_wh: Tuple[int, int],
    keep_aspect_ratio: bool = True,
) -> Float[np.ndarray, "n 2"]:
    """
    Converts dst image pixel coordinates to src image pixel coordinates, where
    the src image is reshaped to the dst image.

    Args:
        dst_pixels: Numpy array of shape (N, 2), each row is (col, row) index.
        src_wh: The size of src image (width, height). Be careful with the order.
        dst_wh: The size of dst image (width, height). Be careful with the order.
        keep_aspect_ratio: Whether the aspect ratio is kept during resizing. If
            True, the src image is reshaped to the dst image with possible
            paddings to keep the aspect ratio. If False, the src image is
            directly reshaped to the dst image. If False,

    Returns:
        src_pixels: Numpy array of shape (N, 2), each row is (col, row) index.
            The coordinates will not be rounded to integers. Out-of-bound values
            will not be corrected.

    Notes:
        1. This function is paired with OpenCV's cv2.resize() function, where
           the *center* of the top-left pixel is considered to be (0, 0).
           - Top-left     corner: (-0.5   , -0.5   )
           - Bottom-right corner: (w - 0.5, h - 0.5)
           However, most other implementations in computer graphics treat the
           *corner* of the top-left pixel to be (0, 0). For more discussions, see:
           https://www.realtimerendering.com/blog/the-center-of-the-pixel-is-0-50-5/
        2. OpenCV's image size is (width, height), while numpy's array shape
           is (height, width) or (height, width, 3). Be careful with the order.
    """
    sanity.assert_shape_nx2(dst_pixels)
    src_w, src_h = src_wh[:2]
    dst_w, dst_h = dst_wh[:2]

    # Compute intermediate shape (tmp_h, tmp_w)
    if not keep_aspect_ratio:
        # Case 1: direct reshape, do not keep aspect ratio.
        tmp_w, tmp_h = dst_w, dst_h
    else:
        # Case 2; keep aspect ratio and fill.
        src_wh_ratio = src_w / float(src_h)
        dst_wh_ratio = dst_w / float(dst_h)
        if src_wh_ratio >= dst_wh_ratio:
            # Source image is "wider". Pad in the height dimension.
            tmp_w = dst_w
            tmp_h = int(round(tmp_w / src_wh_ratio))
        else:
            # Source image is "taller". Pad in the width dimension.
            tmp_h = dst_h
            tmp_w = int(round(tmp_h * src_wh_ratio))
        assert tmp_w <= dst_w and tmp_h <= dst_h

    # Mapping relationship, linear interpolate between:
    # src                                 -> tmp
    # src_tl = (-0.5 , -0.5)              -> dst_tl = (-0.5 , -0.5)
    # src_br = (src_w - 0.5, src_h - 0.5) -> dst_br = (tmp_w - 0.5, tmp_h - 0.5)
    #
    # dst_pixels - dst_tl    src_pixels - src_tl
    # ------------------- == --------------------
    # dst_br - dst_tl        src_br - src_tl
    src_tl = np.array([-0.5, -0.5])
    src_br = np.array([src_w - 0.5, src_h - 0.5])
    dst_tl = np.array([-0.5, -0.5])
    dst_br = np.array([tmp_w - 0.5, tmp_h - 0.5])
    src_pixels = (dst_pixels - dst_tl) / (dst_br - dst_tl) * (src_br - src_tl) + src_tl

    return src_pixels


def make_corres_image(
    im_src: Float[np.ndarray, "h w 3"],
    im_dst: Float[np.ndarray, "h w 3"],
    src_pixels: Int[np.ndarray, "n 2"],
    dst_pixels: Int[np.ndarray, "n 2"],
    confidences: Optional[Float[np.ndarray, "n"]] = None,
    texts: Optional[List[str]] = None,
    point_color: Optional[Tuple[float, ...]] = (0, 1, 0, 1.0),
    line_color: Optional[Tuple[float, ...]] = (0, 0, 1, 0.75),
    text_color: Tuple[float, float, float] = (1, 1, 1),
    point_size: int = 1,
    line_width: int = 1,
    sample_ratio: Optional[float] = None,
) -> Float[np.ndarray, "h 2*w 3"]:
    """
    Make correspondence image.

    Args:
        im_src: (h, w, 3) float image, range 0-1.
        im_dst: (h, w, 3) float image, range 0-1.
        src_pixels: (n, 2) int array, each row represents (x, y) or (c, r).
        dst_pixels: (n, 2) int array, each row represents (x, y) or (c, r).
        confidences: (n,) float array, confidence of each corres, range [0, 1].
        texts: List of texts to draw on the top-left of the image.
        point_color: RGB or RGBA color of the point, float, range 0-1.
            - If point_color == None:
                points will never be drawn.
            - If point_color != None and confidences == None
                point color will be determined by point_color.
            - If point_color != None and confidences != None
                point color will be determined by "viridis" colormap.
        line_color: RGB or RGBA color of the line, float, range 0-1.
        text_color: RGB color of the text, float, range 0-1.
        point_size: Size of the point.
        line_width: Width of the line.
        sample_ratio: Float value from 0-1. If None, all points are drawn.

    Returns:
        Correspondence image with shape (h, 2*w, 3).
    """
    assert im_src.shape == im_dst.shape
    assert im_src.ndim == 3 and im_src.shape[2] == 3
    assert im_src.dtype == np.float32 or im_src.dtype == np.float64
    assert im_dst.dtype == np.float32 or im_dst.dtype == np.float64
    assert im_src.min() >= 0.0 and im_src.max() <= 1.0
    assert im_dst.min() >= 0.0 and im_dst.max() <= 1.0

    assert src_pixels.shape == dst_pixels.shape
    assert src_pixels.ndim == 2 and src_pixels.shape[1] == 2
    assert src_pixels.dtype == np.int32 or src_pixels.dtype == np.int64
    assert dst_pixels.dtype == np.int32 or dst_pixels.dtype == np.int64
    assert len(src_pixels) == len(dst_pixels)

    if confidences is not None:
        assert len(confidences) == len(src_pixels)
        assert confidences.dtype == np.float32 or confidences.dtype == np.float64
        if confidences.size > 0:
            assert confidences.min() >= 0.0 and confidences.max() <= 1.0
        assert confidences.ndim == 1

    # Get shape.
    h, w, _ = im_src.shape

    # Sample corres.
    sample_ratio = 1.0 if sample_ratio is None else sample_ratio
    if sample_ratio > 1.0 or sample_ratio < 0.0:
        raise ValueError("sample_ratio should be in [0.0, 1.0]")
    elif sample_ratio == 1.0:
        pass
    else:
        n = src_pixels.shape[0]
        n_sample = int(round(n * sample_ratio))
        idx = np.random.choice(n, n_sample, replace=False)
        src_pixels = src_pixels[idx]
        dst_pixels = dst_pixels[idx]

    # If there is no corres, return the original images side by side.
    if len(src_pixels) == 0:
        im_corres = np.concatenate((im_src, im_dst), axis=1)

    else:
        assert src_pixels[:, 0].min() >= 0 and src_pixels[:, 0].max() < w
        assert src_pixels[:, 1].min() >= 0 and src_pixels[:, 1].max() < h
        assert dst_pixels[:, 0].min() >= 0 and dst_pixels[:, 0].max() < w
        assert dst_pixels[:, 1].min() >= 0 and dst_pixels[:, 1].max() < h

        # Sanity check: point_color and line_color.
        if point_color is not None:
            assert len(point_color) in {3, 4}
        if line_color is not None:
            assert len(line_color) in {3, 4}

        # Concatenate images.
        im_corres = np.concatenate((im_src, im_dst), axis=1)

        # Sample corres.
        if sample_ratio is not None:
            assert sample_ratio > 0.0 and sample_ratio <= 1.0
            num_points = len(src_pixels)
            num_samples = int(round(num_points * sample_ratio))
            sample_indices = np.random.choice(num_points, num_samples, replace=False)
            src_pixels = src_pixels[sample_indices]
            dst_pixels = dst_pixels[sample_indices]
            confidences = confidences[sample_indices]

        # Draw points.
        if point_color is not None:
            assert len(point_color) == 4 or len(point_color) == 3
            assert np.min(point_color) >= 0.0 and np.max(point_color) <= 1.0

            if confidences is None:
                # Draw white points as mask.
                im_point_mask = np.zeros(im_corres.shape[:2], dtype=im_corres.dtype)
                for (src_c, src_r), (dst_c, dst_r) in zip(src_pixels, dst_pixels):
                    cv2.circle(
                        im_point_mask,
                        (src_c, src_r),
                        point_size,
                        (1,),
                        -1,
                    )
                    cv2.circle(
                        im_point_mask,
                        (dst_c + w, dst_r),
                        point_size,
                        (1,),
                        -1,
                    )
                point_alpha = point_color[3] if len(point_color) == 4 else 1.0
                point_color = point_color[:3]
                im_corres = overlay_mask_on_rgb(
                    im_corres,
                    im_point_mask,
                    overlay_alpha=point_alpha,
                    overlay_color=point_color,
                )
            else:
                # Query color map for colors, given confidences from 0-1.
                colors = colormap.query(confidences, colormap="viridis")

                # Draw points.
                for (src_c, src_r), (dst_c, dst_r), color in zip(
                    src_pixels, dst_pixels, colors
                ):
                    cv2.circle(
                        im_corres,
                        (src_c, src_r),
                        point_size,
                        tuple(color.tolist()),
                        -1,
                    )
                    cv2.circle(
                        im_corres,
                        (dst_c + w, dst_r),
                        point_size,
                        tuple(color.tolist()),
                        -1,
                    )

        # Draw lines.
        if line_color is not None:
            assert len(line_color) == 4 or len(line_color) == 3
            assert np.min(line_color) >= 0.0 and np.max(line_color) <= 1.0

            # Draw white lines as mask.
            im_line_mask = np.zeros(im_corres.shape[:2], dtype=im_corres.dtype)
            for (src_c, src_r), (dst_c, dst_r) in zip(src_pixels, dst_pixels):
                cv2.line(
                    im_line_mask, (src_c, src_r), (dst_c + w, dst_r), (1,), line_width
                )

            line_alpha = line_color[3] if len(line_color) == 4 else 1.0
            line_color = line_color[:3]
            im_corres = overlay_mask_on_rgb(
                im_corres,
                im_line_mask,
                overlay_alpha=line_alpha,
                overlay_color=line_color,
            )

    # Draw texts.
    if texts:

        def get_scales(im_height, max_lines, font, line_text_h_ratio):
            (_, text_h), _ = cv2.getTextSize("ABCDE", font, 1, 1)
            text_h = text_h * line_text_h_ratio
            expected_text_h = im_height / max_lines

            font_scale = int(round(expected_text_h / text_h))
            (_, text_h), _ = cv2.getTextSize("ABCDE", font, font_scale, 1)
            line_h = text_h * line_text_h_ratio
            line_h = int(round(line_h))
            text_h = int(round(text_h))

            return font_scale, line_h, text_h

        font = cv2.FONT_HERSHEY_DUPLEX
        line_text_h_ratio = 1.2
        max_lines = 20
        font_scale, line_h, text_h = get_scales(
            im_corres.shape[0], max_lines, font, line_text_h_ratio
        )
        font_thickness = 2
        org = (line_h, line_h * 2)

        for text in texts:
            im_corres = cv2.putText(
                im_corres,
                text,
                org,
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            org = (org[0], org[1] + line_h)

    assert im_corres.min() >= 0.0 and im_corres.max() <= 1.0

    return im_corres


def vstack_images(
    ims: List[Float[np.ndarray, "h w 3"]],
    alignment: str = "left",
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Float[np.ndarray, "h_stacked w_stacked 3"]:
    """
    Vertically stacks images, aligning them to "left", "center", or "right",
    with a specified background color.

    Args:
        ims (List[np.ndarray]): List of float32/float64 images with shape
            (H, W, 3) and pixel values in [0.0, 1.0].
        alignment (str): How to align images ("left", "center", "right").
            Defaults to "left".
        background_color (Tuple[float, float, float]): The background color of
            the stacked image, specified as a tuple of three floats (R, G, B)
            in the range [0.0, 1.0]. Defaults to white (1.0, 1.0, 1.0).

    Returns:
        np.ndarray: Stacked image as float32.
    """
    for im in ims:
        if im.ndim != 3 or im.shape[2] != 3:
            raise ValueError("Each image must be 3D with 3 channels.")
        if im.dtype not in [np.float32, np.float64]:
            raise ValueError("Image dtype must be float32/float64.")
        if im.min() < 0.0 or im.max() > 1.0:
            raise ValueError("Pixels must be in [0.0, 1.0].")
    if not all(0 <= c <= 1 for c in background_color):
        raise ValueError(
            f"background_color must be 3 floats in the range [0, 1], "
            f"but got {background_color}."
        )
    valid_alignments = ["left", "center", "right"]
    if alignment not in valid_alignments:
        raise ValueError(
            f"Invalid alignment: '{alignment}', must be one of {valid_alignments}."
        )

    max_width = max(im.shape[1] for im in ims)
    total_height = sum(im.shape[0] for im in ims)

    im_stacked = np.ones((total_height, max_width, 3), dtype=np.float32)
    im_stacked = im_stacked * np.array(background_color).reshape(1, 1, 3)

    curr_row = 0
    for im in ims:
        offset = (
            (max_width - im.shape[1]) // 2
            if alignment == "center"
            else max_width - im.shape[1] if alignment == "right" else 0
        )
        im_stacked[curr_row : curr_row + im.shape[0], offset : offset + im.shape[1]] = (
            im
        )
        curr_row += im.shape[0]

    return im_stacked
