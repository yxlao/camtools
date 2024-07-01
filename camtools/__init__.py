from . import artifact
from . import backend
from . import camera
from . import colmap
from . import colormap
from . import convert
from . import geometry
from . import image
from . import io
from . import metric
from . import normalize
from . import project
from . import raycast
from . import render
from . import sanity
from . import solver
from . import transform
from . import typing
from . import util

try:
    # Python >= 3.8
    from importlib.metadata import version

    __version__ = version("camtools")
except ImportError:
    # Python < 3.8
    import pkg_resources

    __version__ = pkg_resources.get_distribution("camtools").version
