import cv2
from cv2 import CV_32F
import numpy as np
from pathlib import Path


def imwrite(im_path, im):
    """
    Write image, with no surprises.

    Args:
        im_path: Path to image.
            - Only ".jpg" or ".png" is supported.
            - Directory will be created if it does not exist.
        im: Numpy array.
            - ndims: Must be 1 or 3. Otherwise, an exception will be thrown.
            - dtype: Must be uint8, float32, float64.

    Notes:
        - You should not use this to save a depth image (typically uint16).
        - Float image will get a range check to ensure it is in [0, 1].
    """
    im_path = Path(im_path)

    assert im_path.suffix in (".jpg", ".png")
    assert isinstance(im, np.ndarray)
    assert im.ndim in (1, 3)
    assert im.dtype in [np.uint8, np.float32, np.float64]

    # Float to uint8.
    if im.dtype == np.float32 or im.dtype == np.float64:
        im_min = im.min()
        im_max = im.max()
        if im_max > 1 or im_min < 0:
            raise ValueError(f"Image out-of-range: min {im_min} max {im_max}.")
        # Should we use round()?
        im = (im * 255).round().astype(np.uint8)
    elif im.dtype == np.uint8:
        pass
    else:
        raise ValueError(f"Unsupported image type: {im.dtype}")

    # RGB to BGR.
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    # Write.
    im_dir = im_path.parent
    im_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(im_path), im)


def imwrite_depth(im_path, im, depth_scale=1000.0):
    """
    Multiply depths by depth_scale and write depth image to a 16-bit .png file.

    Args:
        im_path: Path to image. Must be "*.png". Folders will be created if
            necessary.
        im: Numpy array. Must be float32 or float64.

    Note:
        The user is responsible for defining what is invalid depth. E.g.,
        invalid depth can represented as np.nan, np.inf, 0, -1, etc. This
        function simply multiplies the depth by depth_scale can convert to
        uint16. For instance, with depth_scale = 1000,
            - Input depths     : [np.nan, np.inf, -np.inf,   0,      -1,   3.14]
            - Written to ".png": [     0,      0,       0,   0,   64536,   3140]
            - Read from ".png" : [     0,      0,       0,   0,   64536,   3140]
            - Convert to float : [     0,      0,       0,   0,  64.536,   3.14]
                                                             ^
                                                        Best practice.
        Note that -1 is converted to 64536 / 1000 = 64.536 meters, therefore,
        it is important to clip depth with min_depth and max_depth. The best
        practice is to use 0 as invalid depth.
    """
    im_path = Path(im_path)

    assert im_path.suffix == ".png"
    assert isinstance(im, np.ndarray)
    assert im.dtype in [np.float32, np.float64]
    assert im.ndim == 2

    im = (im * depth_scale).astype(np.uint16)

    im_dir = im_path.parent
    im_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(im_path), im)


def imread(im_path, alpha_mode=None):
    """
    Read image, with no surprises.
    - Input : uint8 (divide by 255) or uint16 (divide by 65535) image.
              If you're reading a depth uint16 image, use imread_depth() instead.
    - Return: float32, RGB, range [0, 1] image will be returned.

    Args:
        im_path: Path to image.
        alpha_mode: Specifies how to handle alpha channel.
            - None    : Default. Throw an exception if alpha channel is present.
            - "keep"  : Keep the alpha channel if presented. Returns an RGB or
                        an RGBA image.
            - "ignore": Ignore alpha channel. Returns an RGB image.
            - "white" : Fill with white background. Returns an RGB image.
            - "black" : Fill with black background. Returns an RGB image.
    Returns:
        An image in float32, and range from 0 to 1. Possible number of channels:
        - alpha_mode == None    : {1, 3}
        - alpha_mode == "keep"  : {1, 3, 4}
        - alpha_mode == "ignore": {1, 3}
        - alpha_mode == "white" : {1, 3}
        - alpha_mode == "black" : {1, 3}

    Notes:
        - If the image has 3 channels, the order will be R, G, B.
        - If image dtype is uint16, an exception will be thrown.
    """
    im_path = Path(im_path)
    assert im_path.suffix in (".jpg", ".png")
    assert im_path.is_file(), f"{im_path} is not a file."

    # Read.
    im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)

    # Handle dtypes.
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    elif im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    else:
        raise ValueError(f"Unsupported image dtype: {im.dtype}")

    # Handle channels.
    if im.ndim == 2:
        pass
    elif im.ndim == 3:
        if im.shape[2] == 4:
            if alpha_mode is None:
                raise ValueError(
                    f"{im_path} has an alpha channel, alpha_mode " f"must be specified."
                )
            elif alpha_mode == "keep":
                pass
            elif alpha_mode == "ignore":
                im = im[:, :, :3]
            elif alpha_mode == "white":
                # (H, W, 1).
                alpha = im[..., 3:]
                # (H, W, 3).
                foreground = im[..., :3] * alpha
                # (H, W, 1).
                background = (1.0 - alpha) * 1.0
                # (H, W, 3), background is broadcasted.
                im = foreground + background
            elif alpha_mode == "black":
                im = im[..., :3] * im[..., 3:]
            else:
                raise ValueError(
                    f"Unexpected alpha_mode: {alpha_mode} for a " "4-channel image."
                )
        elif im.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {im.shape}")

        # BGR to RGB.
        # Now, im.shape[2] can be either 3 or 4.
        if im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        elif im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected image shape: {im.shape}")
    else:
        raise ValueError(f"Unexpected image shape: {im.shape}")

    # Sanity check.
    if im.dtype != np.float32:
        raise ValueError(f"Internal Error. Image must float32.")

    # This can be avoided.
    if im.min() > 1 or im.max() < 0:
        raise ValueError(
            f"Internal Error. Image must be in range [0, 1], but "
            f"got [{im.min()}, {im.max()}]"
        )

    return im


def imread_depth(im_path, depth_scale=1000.0):
    """
    Read depth image from a 16-bit .png file and divide depths by depth_scale.

    Args:
        im_path: Path to image. Must be "*.png".

    Returns:
        Numpy array with dtype float32.

    Note:
        The user is responsible for defining what is invalid depth. E.g.,
        invalid depth can represented as np.nan, np.inf, 0, -1, etc. This
        function simply multiplies the depth by depth_scale can convert to
        uint16. For instance, with depth_scale = 1000,
            - Input depths     : [np.nan, np.inf, -np.inf,   0,      -1,   3.14]
            - Written to ".png": [     0,      0,       0,   0,   64536,   3140]
            - Read from ".png" : [     0,      0,       0,   0,   64536,   3140]
            - Convert to float : [     0,      0,       0,   0,  64.536,   3.14]
                                                             ^
                                                        Best practice.
        Note that -1 is converted to 64536 / 1000 = 64.536 meters, therefore,
        it is important to clip depth with min_depth and max_depth. The best
        practice is to use 0 as invalid depth.
    """
    im_path = Path(im_path)
    assert im_path.suffix == ".png"
    assert im_path.is_file(), f"{im_path} is not a file."

    im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)

    if im.dtype != np.uint16 and im.dtype != np.uint8:
        raise ValueError(f"Unexpected image type: {im.dtype}")

    im = im.astype(np.float32) / depth_scale

    return im
