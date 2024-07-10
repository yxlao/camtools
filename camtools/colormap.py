import numpy as np
from . import io

from .backend import Tensor, tensor_backend_numpy, tensor_backend_auto, ivy
from jaxtyping import Float
from typing import List, Tuple, Dict
from matplotlib import pyplot as plt


@tensor_backend_numpy
def query(
    values: Float[Tensor, "..."],
    colormap="viridis",
) -> Float[Tensor, "... 3"]:
    """
    Query matplotlib's color map.

    Args:
        values: Numpy array in float32 or float64. Valid range is [0, 1]. It
            can be of arbitrary shape. As matplotlib color maps have different
            behaviors for float and int, the input array should be in float.
        colormap: Name of matplotlib color map.

    Returns:
        Numpy array of shape (**values.shape, 3) with dtype float32.
    """
    try:
        cmap = plt.get_cmap(colormap)
    except AttributeError:
        cmap = plt.colormaps.get_cmap(colormap)

    colors = cmap(values)
    colors = colors[..., :3]  # Remove alpha channel if present

    return colors.astype(np.float32)


@tensor_backend_auto
def normalize(
    array: Float[Tensor, "..."],
    vmin: float = 0.0,
    vmax: float = 1.0,
    clip: bool = False,
):
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
        array = ivy.clip(array, x_min=vmin, x_max=vmax)
    else:
        amin = array.min()
        amax = array.max()
        array = (array - amin) / (amax - amin) * (vmax - vmin) + vmin

    return array
