import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional
from jaxtyping import UInt8, Float


def is_jpg_path(path):
    return Path(path).suffix.lower() in [".jpg", ".jpeg"]


def is_png_path(path):
    return Path(path).suffix.lower() in [".png"]


def imwrite(
    im_path: Union[str, Path],
    im: Union[
        UInt8[np.ndarray, "h w"],
        UInt8[np.ndarray, "h w 3"],
        Float[np.ndarray, "h w"],
        Float[np.ndarray, "h w 3"],
    ],
    quality: int = 95,
) -> None:
    """
    Write an image, with no surprises.

    This function handles common image writing tasks including:
    - Automatic directory creation
    - Image format detection based on file extension
    - Input validation and type conversion
    - Color space conversion (RGB to BGR for OpenCV)

    Args:
        im_path (Union[str, Path]): Path to save the image. Supported extensions:
            - .jpg/.jpeg: JPEG format with configurable quality
            - .png: PNG format with lossless compression
            Note: Parent directories will be created automatically if they don't exist.

        im (Union[UInt8[np.ndarray], Float[np.ndarray]]): Image data as a numpy array.
            Supported formats:
            - Grayscale: 2D array (height x width)
            - Color: 3D array (height x width x 3)
            Supported data types:
            - uint8: Values in range [0, 255]
            - float32/float64: Values in range [0.0, 1.0] (will be scaled to uint8)

        quality (int, optional): JPEG quality setting (1-100). Defaults to 95.
            Higher values mean better quality but larger file size. Only applies
            to JPEG images.

    Notes:
        - Depth images (typically uint16) should use imwrite_depth() instead
        - Float images are automatically scaled to [0, 1] range and converted to uint8
        - For PNG images, the quality parameter is ignored
        - Color images are automatically converted from RGB to BGR format for OpenCV

    Examples:
        >>> # Save a grayscale image
        >>> grayscale = np.random.rand(256, 256).astype(np.float32)
        >>> imwrite('output.jpg', grayscale)

        >>> # Save a color image with high quality
        >>> color = np.random.rand(256, 256, 3).astype(np.uint8)
        >>> imwrite('output.jpg', color, quality=100)
    """
    im_path = Path(im_path)

    assert is_jpg_path(im_path) or is_png_path(
        im_path
    ), f"{im_path} is not a JPG or PNG file."
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
    if is_jpg_path(im_path):
        cv2.imwrite(str(im_path), im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(im_path), im)


def imwrite_depth(
    im_path: Union[str, Path],
    im: Float[np.ndarray, "h w"],
    depth_scale: float = 1000.0,
) -> None:
    """
    Write depth map to a 16-bit PNG file with depth scaling.

    This function handles depth map storage by:
    - Scaling depth values by depth_scale
    - Converting to 16-bit unsigned integer format
    - Creating necessary directories
    - Validating input data

    Args:
        im_path (Union[str, Path]): Output file path. Must have .png extension.
            Parent directories will be created automatically if they don't exist.
        im (Float[np.ndarray]): Depth map as a 2D numpy array. Must be:
            - Shape: (height, width)
            - Data type: float32 or float64
            - Values: Depth values in meters (or other consistent units)
        depth_scale (float, optional): Scaling factor to apply before converting
            to uint16. Defaults to 1000.0. This determines the precision of
            stored depth values. For example:
            - depth_scale=1000: 1mm precision
            - depth_scale=100: 1cm precision
            - depth_scale=1: 1m precision

    Notes:
        - Invalid depth values (np.nan, np.inf, etc.) are converted to 0
        - Depth values are clipped to uint16 range (0-65535) after scaling
        - For best results, use 0 to represent invalid depth values
        - When reading the depth map with imread_depth(), use the same depth_scale
          to recover the original depth values

        The user is responsible for defining what is invalid depth. For example,
        invalid depth can be represented as np.nan, np.inf, 0, -1, etc. This
        function simply multiplies the depth by depth_scale and converts to
        uint16. For instance, with depth_scale = 1000:

        - Input depths     : [np.nan, np.inf, -np.inf,   0,      -1,   3.14]
        - Written to ".png": [     0,      0,       0,   0,   64536,   3140]
        - Read from ".png" : [     0,      0,       0,   0,   64536,   3140]
        - Convert to float : [     0,      0,       0,   0,  64.536,   3.14]

        Note that -1 is converted to 64536 / 1000 = 64.536 meters, therefore,
        it is important to clip depth with min_depth and max_depth. The best
        practice is to use 0 as invalid depth.

    Examples:
        >>> # Write depth map with 1mm precision
        >>> depth = np.random.rand(256, 256).astype(np.float32) * 10  # 0-10 meters
        >>> imwrite_depth('depth.png', depth, depth_scale=1000)

        >>> # Write depth map with 1cm precision
        >>> imwrite_depth('depth.png', depth, depth_scale=100)
    """
    im_path = Path(im_path)

    assert is_png_path(im_path), f"{im_path} is not a PNG file."
    assert isinstance(im, np.ndarray)
    assert im.dtype in [np.float32, np.float64]
    assert im.ndim == 2

    im = (im * depth_scale).astype(np.uint16)

    im_dir = im_path.parent
    im_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(im_path), im)


def imread(
    im_path: Union[str, Path],
    alpha_mode: Optional[str] = None,
) -> Union[
    Float[np.ndarray, "h w"],
    Float[np.ndarray, "h w 3"],
    Float[np.ndarray, "h w 4"],
]:
    """
    Read and image, with no surprises.
    Guaranteed to return float32 arrays in range [0, 1] with RGB(A) color space.

    This function handles image reading by:
    - Automatically converting to float32 in range [0, 1]
    - Supporting both grayscale and color images
    - Providing multiple options for handling alpha channels
    - Converting color space from BGR to RGB

    Args:
        im_path (Union[str, Path]): Path to the image file. Supported formats:
            - .jpg/.jpeg: JPEG format
            - .png: PNG format (may contain alpha channel)

        alpha_mode (Optional[str]): Specifies how to handle alpha channels:
            - None    : Default. Raise error if alpha channel is present
            - "keep"  : Preserve alpha channel (returns RGBA)
            - "ignore": Discard alpha channel (returns RGB)
            - "white" : Composite with white background (returns RGB)
            - "black" : Composite with black background (returns RGB)

    Returns:
        Union[Float[np.ndarray]]: Normalized image array with:
            - Data type: float32
            - Value range: [0, 1]
            - Possible shapes:
                * Grayscale: (height, width)
                * Color: (height, width, 3)
                * With alpha: (height, width, 4) (only when alpha_mode="keep")
            - Possible number of channels based on alpha_mode:
                - alpha_mode == None    : {1, 3}
                - alpha_mode == "keep"  : {1, 3, 4}
                - alpha_mode == "ignore": {1, 3}
                - alpha_mode == "white" : {1, 3}
                - alpha_mode == "black" : {1, 3}

    Notes:
        - Input images must be uint8 or uint16 format
        - Color images are automatically converted from BGR to RGB
        - For depth images, use imread_depth() instead
        - When alpha_mode is None, images with alpha channels will raise an error

    Examples:
        >>> # Read grayscale image
        >>> gray = imread('image.jpg')

        >>> # Read color image, ignore alpha if present
        >>> rgb = imread('image.png', alpha_mode='ignore')

        >>> # Read image with alpha channel preserved
        >>> rgba = imread('image.png', alpha_mode='keep')

        >>> # Read image with white background for transparency
        >>> rgb_white = imread('image.png', alpha_mode='white')
    """
    im_path = Path(im_path)
    assert is_jpg_path(im_path) or is_png_path(
        im_path
    ), f"{im_path} is not a JPG or PNG file."
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
                    f"{im_path} has an alpha channel, alpha_mode "
                    f"must be specified."
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
                    f"Unexpected alpha_mode: {alpha_mode} for a "
                    "4-channel image."
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


def imread_depth(
    im_path: Union[str, Path],
    depth_scale: float = 1000.0,
) -> Float[np.ndarray, "h w"]:
    """
    Read a depth map from a 16-bit PNG file and convert to float.

    This function handles depth map reading by:
    - Loading 16-bit PNG data
    - Converting to float32 format
    - Applying depth scale to recover original values
    - Validating input data

    Args:
        im_path (Union[str, Path]): Path to the depth map PNG file.
        depth_scale (float, optional): Scaling factor to convert from uint16 to
            float. Defaults to 1000.0. Must match the scale used when saving
            the depth map. For example:
            - depth_scale=1000: 1mm precision
            - depth_scale=100: 1cm precision
            - depth_scale=1: 1m precision

    Returns:
        Float[np.ndarray, "h w"]: Depth map as a float32 array with shape
            (height, width). Values are in the original units (typically meters).

    Notes:
        - Zero values in the PNG file are preserved as zeros in the output
        - Non-zero values are divided by depth_scale to recover original depths
        - Use the same depth_scale value that was used with imwrite_depth()
        - For best results, use 0 to represent invalid depth values

    Examples:
        >>> # Read depth map saved with 1mm precision
        >>> depth = imread_depth('depth.png', depth_scale=1000)

        >>> # Read depth map saved with 1cm precision
        >>> depth = imread_depth('depth.png', depth_scale=100)
    """
    im_path = Path(im_path)
    assert is_png_path(im_path), f"{im_path} is not a PNG file."
    assert im_path.is_file(), f"{im_path} is not a file."

    im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)

    if im.dtype != np.uint16 and im.dtype != np.uint8:
        raise ValueError(f"Unexpected image type: {im.dtype}")

    im = im.astype(np.float32) / depth_scale

    return im
