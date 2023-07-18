import matplotlib
import numpy as np
from . import io


def query(points, colormap="viridis"):
    """
    Query matplotlib's color map.

    Args:
        points: Numpy array in float32 or float64. Valid range is [0, 1].
        colormap: Name of matplotlib color map.

    Returns:
        Numpy array of shape (**points.shape, 3) with dtype float32.
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


def normalize(array, vmin=0.0, vmax=1.0, clip=False):
    """
    Normalize array to [vmin, vmax].

    Args:
        array: Numpy array.
        vmin: Minimum value.
        vmax: Maximum value.
        clip: If True, clip array to [vmin, vmax].

    Returns:
        Normalized array of the same shape as the input array.
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
