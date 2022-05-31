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
            - ndims: Must be 1 or 3. When n
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


def imread(im_path):
    """
    Read image, with no surprises. Float32 image [0, 1] will be returned.

    Args:
        im_path: Path to image.

    Returns:
        Float32 image with range from 0 to 1.

    Notes:
        - If image dtype is uint16, an exception will be thrown.
        - Alpha channel will be ignored, a warning will be printed.
    """
    im_path = Path(im_path)
    assert im_path.suffix in (".jpg", ".png")
    assert im_path.is_file()

    # Read.
    im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)

    # Check dtype.
    if im.dtype == np.uint8:
        pass
    elif im.dtype == np.uint16:
        raise ValueError(f"Unsupported image type: {im.dtype}. Please use:\n"
                         f"cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)")
    else:
        raise ValueError(f"Unexpected image type: {im.dtype}")

    # Handle channels.
    if im.ndim == 1:
        pass
    elif im.ndim == 3:
        # Ignore alpha channel.
        if im.shape[2] == 4:
            print(f"Warning: alpha channel ignored.")
            im = im[:, :, :3]
        elif im.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {im.shape}")
        # BGR to RGB.
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unexpected image shape: {im.shape}")

    # Handle dtypes.
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    elif im.dtype == np.float32 or im.dtype == np.float64:
        # We shouldn't reach here. Do a sanity check anyway.
        im_min = im.min()
        im_max = im.max()
        if im_max > 1 or im_min < 0:
            raise ValueError(f"Image out-of-range: min {im_min} max {im_max}.")
        pass
    else:
        raise ValueError(f"Unsupported image type: {im.dtype}")

    return im
