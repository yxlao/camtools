import logging

from . import artifact
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
from . import util


# Get package version for camtools
try:
    # Python >= 3.8
    from importlib.metadata import version

    def _get_package_version(package):
        return version(package)

except ImportError:
    # Python < 3.8
    import pkg_resources

    def _get_package_version(package):
        return pkg_resources.get_distribution(package).version


__version__ = _get_package_version("camtools")


# Check open3d and numpy compatibility
# https://github.com/isl-org/Open3D/issues/6840
try:
    logging.basicConfig(format="%(message)s")
    _logger = logging.getLogger(__name__)

    o3d_version = _get_package_version("open3d")
    np_version = _get_package_version("numpy")

    o3d_version_tuple = tuple(map(int, o3d_version.split(".")[:2]))
    np_version_tuple = tuple(map(int, np_version.split(".")[:2]))

    if o3d_version_tuple < (0, 19) and np_version_tuple >= (2, 0):
        _logger.warning(
            f"[Warning] Incompatible versions: open3d {o3d_version} does "
            f"not support numpy {np_version}. You may upgrade open3d to >= 0.19.0 "
            f"or downgrade numpy to 1.x."
        )
except Exception as e:
    pass
