import numpy as np
import cv2
from . import sanity


def overlay_mask_on_rgb(im_rgb,
                        im_mask,
                        overlay_alpha=0.4,
                        overlay_color=[0, 0, 1]):
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


def dst_pixels_to_src_pixels(dst_pixels,
                             src_wh,
                             dst_wh,
                             keep_aspect_ratio=True):
    """
    Converts dst image pixel coordinates to src image pixel coordinates, where
    the src image is reshaped to the dst image.

    Args:
        dst_pixels: Numpy array of shape (N, 2), each row is (col, row) index.
        src_wh: The size of src image (width, height). Be careful with the order.
        dst_wh: The size of dst image (width, height). Be careful with the order.
        keep_aspect_ratio: Whether the aspect ratio is kept during resizing. If
            True, the src image is directly reshaped to the dst image. If False,
            the src image is reshaped to the dst image with possible paddings to
            keep the aspect ratio.

    Returns:
        src_pixels: Numpy array of shape (N, 2), each row is (col, row) index.
            The coordinates will not be rounded to integers. Out-of-bound values
            will not be corrected.

    Note:
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
    assert dst_pixels.ndim == 2
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
    src_pixels = (dst_pixels - dst_tl) / (dst_br - dst_tl) * (src_br -
                                                              src_tl) + src_tl

    return src_pixels


def ndc_coords_to_pixels(ndc_coords, im_size_wh, align_corners=False):
    """
    Convert NDC coordinates (from -1 to 1) to pixel coordinates. Out-of-bound
    values will NOT be corrected.

    Args:
        ndc_coords: NDC coordinates. (N, 2) array of floats. Each row represents
            (x, y) or (c, r). Most values shall be in [-1, 1], where (-1, -1) is
            the top left corner and (1, 1) is the bottom right corner.
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
        Pixel coordinates. (N, 2) array of floats.
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

    dst_pixels = (ndc_coords - src_tl) / (src_br - src_tl) * (dst_br -
                                                              dst_tl) + dst_tl

    return dst_pixels


def resize(im, shape_wh, aspect_ratio_fill=None):
    """
    Resize image to shape_wh = (width, height).
    In numpy, the resulting shape is (height, width) or (height, width, 3).

    Args:
        im: Numpy shape {(h, w), (h, w, 3}; dtype {uint8, float32, float64}.
        shape_wh: Tuple of (width, height). Be careful with the order.
        aspect_ratio_fill: The value to fill in order to keep the aspect ratio.
            - If None, image will be directly resized to (height, width).
            - If not None, the number of elements must match the channel size,
              1 or 3. The dtype and range must also match the input image.
              These value(s) will be filled to maintain the aspect ratio.

    Returns:
        im_resized: image of shape (height, width) or (height, width, 3).

    Notes:
        OpenCV's image size is (width, height), while numpy's array shape is
        (height, width) or (height, width, 3). Be careful with the order.
    """
    # Sanity: dtype.
    dtype = im.dtype
    ndim = im.ndim
    if dtype == np.uint8:
        pass
    elif dtype == np.float32 or dtype == np.float64:
        assert im.max() <= 1.0 and im.min() >= 0.0
    else:
        raise ValueError(f"Unsupported dtype {dtype}.")

    # Sanity: input shape.
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
    im_tmp = cv2.resize(im, (tmp_w, tmp_h))

    # Pad if necessary.
    if tmp_w == dst_w and tmp_h == dst_h:
        im_resize = im_tmp
    else:
        im_resize = np.full(dst_numpy_shape,
                            fill_value=aspect_ratio_fill,
                            dtype=dtype)
        im_resize[:tmp_h, :tmp_w] = im_tmp

    # Final sanity checks for the reshaped image.
    assert im_resize.shape == dst_numpy_shape

    return im_resize


def make_corres_image(im_src,
                      im_dst,
                      src_pixels,
                      dst_pixels,
                      texts=None,
                      point_color=(0, 1, 0, 1.0),
                      line_color=(0, 0, 1, 0.75),
                      point_size=1,
                      line_width=1):
    """
    Make correspondence image.

    Args:
        im_src: (h, w, 3) float image, range 0-1.
        im_dst: (h, w, 3) float image, range 0-1.
        src_pixels: (n, 2) int array, each row represents (x, y) or (c, r).
        dst_pixels: (n, 2) int array, each row represents (x, y) or (c, r).
        texts: List of texts to draw on the top-left of the image.
        point_color: RGB or RGBA color of the point, float, range 0-1.
        line_color: RGB or RGBA color of the line, float, range 0-1.
        point_size: Size of the point.
        line_width: Width of the line.
    """
    assert im_src.shape == im_dst.shape
    assert im_src.ndim == 3 and im_src.shape[2] == 3
    assert im_src.dtype == np.float32 or im_src.dtype == np.float64
    assert im_dst.dtype == np.float32 or im_dst.dtype == np.float64
    assert im_src.min() >= 0.0 and im_src.max() <= 1.0
    assert im_dst.min() >= 0.0 and im_dst.max() <= 1.0
    h, w, _ = im_src.shape

    assert src_pixels.shape == dst_pixels.shape
    assert src_pixels.ndim == 2 and src_pixels.shape[1] == 2
    assert src_pixels.dtype == np.int32 or src_pixels.dtype == np.int64
    assert dst_pixels.dtype == np.int32 or dst_pixels.dtype == np.int64

    # If there is no corres, return the original images side by side.
    if len(src_pixels) == 0:
        return np.concatenate((im_src, im_dst), axis=1)

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

    # Draw points.
    if point_color is not None:
        assert len(point_color) == 4 or len(point_color) == 3
        assert np.min(point_color) >= 0.0 and np.max(point_color) <= 1.0

        # Draw white points as mask.
        im_point_mask = np.zeros(im_corres.shape[:2], dtype=im_corres.dtype)
        for (src_c, src_r), (dst_c, dst_r) in zip(src_pixels, dst_pixels):
            cv2.circle(im_point_mask, (src_c, src_r), point_size, (1,), -1)
            cv2.circle(im_point_mask, (dst_c + w, dst_r), point_size, (1,), -1)

        point_alpha = point_color[3] if len(point_color) == 4 else 1.0
        point_color = point_color[:3]
        im_corres = overlay_mask_on_rgb(im_corres,
                                        im_point_mask,
                                        overlay_alpha=point_alpha,
                                        overlay_color=point_color)

    # Draw lines.
    if line_color is not None:
        assert len(line_color) == 4 or len(line_color) == 3
        assert np.min(line_color) >= 0.0 and np.max(line_color) <= 1.0

        # Draw white lines as mask.
        im_line_mask = np.zeros(im_corres.shape[:2], dtype=im_corres.dtype)
        for (src_c, src_r), (dst_c, dst_r) in zip(src_pixels, dst_pixels):
            cv2.line(im_line_mask, (src_c, src_r), (dst_c + w, dst_r), (1,),
                     line_width)

        line_alpha = line_color[3] if len(line_color) == 4 else 1.0
        line_color = line_color[:3]
        im_corres = overlay_mask_on_rgb(im_corres,
                                        im_line_mask,
                                        overlay_alpha=line_alpha,
                                        overlay_color=line_color)

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
        font_scale, line_h, text_h = get_scales(im_corres.shape[0], max_lines,
                                                font, line_text_h_ratio)
        font_thickness = 2
        font_color = (1, 1, 1)
        org = (line_h, line_h * 2)

        for text in texts:
            im_corres = cv2.putText(im_corres, text, org, font, font_scale,
                                    font_color, font_thickness, cv2.LINE_AA)
            org = (org[0], org[1] + line_h)

    assert im_corres.min() >= 0.0 and im_corres.max() <= 1.0

    return im_corres
