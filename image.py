import numpy as np


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
