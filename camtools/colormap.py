import matplotlib
import numpy as np
from jaxtyping import Float
from . import io


def query(
    values: Float[np.ndarray, "*batch"],
    colormap: str = "viridis",
) -> Float[np.ndarray, "*batch 3"]:
    """
    Query matplotlib's color map.

    Args:
        values: Scalar values to map to colors. Valid range is [0, 1].
        colormap: Name of matplotlib color map.

    Returns:
        RGB colors corresponding to input values.

    Raises:
        ValueError: If values.dtype is not float32 or float64.
    """
    assert isinstance(values, np.ndarray)

    if not values.dtype == np.float32 and not values.dtype == np.float64:
        raise ValueError(
            "Matplotlib's colormap has different behavior for ints and floats. "
            "To unify behavior, we require floats (between 0-1 if valid). "
            f"However, dtype of {values.dtype} is used."
        )

    cmap = matplotlib.cm.get_cmap(colormap)
    colors = cmap(values)[..., :3]  # Remove alpha.

    return colors.astype(np.float32)


def normalize(
    array: Float[np.ndarray, "*batch"],
    vmin: float = 0.0,
    vmax: float = 1.0,
    clip: bool = False,
) -> Float[np.ndarray, "*batch"]:
    """
    Normalize array to [vmin, vmax].

    Args:
        array: Input array to normalize.
        vmin: Minimum value in output range.
        vmax: Maximum value in output range.
        clip: If True, clip array to [vmin, vmax].

    Returns:
        Normalized array with same shape as input.
    """
    if clip:
        array = np.clip(array, vmin, vmax)
    else:
        amin = array.min()
        amax = array.max()
        array = (array - amin) / (amax - amin) * (vmax - vmin) + vmin

    return array
