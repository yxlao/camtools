import matplotlib
import numpy as np
from . import io


def query(points: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """
    Query matplotlib's color map to obtain corresponding colors for given points.

    Args:
        points: Numpy array of type float32 or float64, expected in the range [0, 1].
        colormap: Name of matplotlib color map to use.

    Returns:
        Numpy array of shape (**points.shape, 3) with RGB colors, dtype float32.
        Removes the alpha channel from the colormap output.
    """
    assert isinstance(points, np.ndarray)

    if not points.dtype == np.float32 and not points.dtype == np.float64:
        raise ValueError(
            "Matplotlib's colormap has different behavior for ints and floats. "
            "To unify behavior, we require floats (between 0-1 if valid). "
            f"However, dtype of {points.dtype} is used."
        )

    cmap = matplotlib.cm.get_cmap(colormap)
    colors = cmap(points)[..., :3]  # Remove alpha.

    return colors.astype(np.float32)


def normalize(
    array: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    clip: bool = False,
) -> np.ndarray:
    """
    Normalize array to a specified range [vmin, vmax].

    Args:
        array: Numpy array to be normalized.
        vmin: Minimum value of the target range.
        vmax: Maximum value of the target range.
        clip: If True, values outside [vmin, vmax] are clipped to the range endpoints.

    Returns:
        Normalized array of the same shape as input, scaled to [vmin, vmax].
    """
    if clip:
        array = np.clip(array, vmin, vmax)
    else:
        amin = array.min()
        amax = array.max()
        array = (array - amin) / (amax - amin) * (vmax - vmin) + vmin

    return array


def main():
    """
    Test create color map image.
    """
    height = 200
    width = 1600

    colors = query(np.linspace(0, 1, num=width))
    im = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(width):
        im[:, i : i + 1, :] = colors[i]

    im_path = "colormap.png"
    io.imwrite(im_path, im)


if __name__ == "__main__":
    main()
