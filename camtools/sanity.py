import numpy as np
from typing import Optional
from jaxtyping import Float


def assert_numpy(x, name=None):
    """
    Assert that x is a numpy array.

    Args:
        x: Input to check
        name: Optional name of the variable for error message

    Raises:
        ValueError: If x is not a numpy array
    """
    if not isinstance(x, np.ndarray):
        maybe_name = f" {name}" if name is not None else ""
        raise ValueError(f"Expected{maybe_name} to be numpy array, but got {type(x)}.")


def assert_K(K: Float[np.ndarray, "3 3"]):
    """
    Assert that K is a valid 3x3 camera intrinsic matrix.

    The intrinsic matrix K follows the standard form:
        [[fx,  s, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    where:
        - fx, fy: focal lengths in pixels
        - cx, cy: principal point coordinates
        - s: skew coefficient (usually 0)

    Args:
        K: Camera intrinsic matrix to validate

    Raises:
        ValueError: If K is not a 3x3 matrix
    """
    if K.shape != (3, 3):
        raise ValueError(f"K must has shape (3, 3), but got {K} of shape {K.shape}.")


def assert_T(T: Float[np.ndarray, "4 4"]):
    """
    Assert that T is a valid 4x4 camera extrinsic matrix (world-to-camera transformation).

    The extrinsic matrix T follows the standard form:
        [[R00, R01, R02, t0],
         [R10, R11, R12, t1],
         [R20, R21, R22, t2],
         [  0,   0,   0,  1]]
    where:
        - R is a 3x3 rotation matrix
        - t is a 3x1 translation vector
        - Bottom row must be [0, 0, 0, 1]

    Args:
        T: Camera extrinsic matrix to validate

    Raises:
        ValueError: If T is not a 4x4 matrix or bottom row is not [0, 0, 0, 1]
    """
    if T.shape != (4, 4):
        raise ValueError(f"T must has shape (4, 4), but got {T} of shape {T.shape}.")
    is_valid = np.allclose(T[3, :], np.array([0, 0, 0, 1]))
    if not is_valid:
        raise ValueError(f"T must has [0, 0, 0, 1] the bottom row, but got {T}.")


def assert_pose(pose: Float[np.ndarray, "4 4"]):
    """
    Assert that pose is a valid 4x4 camera pose matrix (camera-to-world transformation).

    The pose matrix follows the standard form:
        [[R00, R01, R02, t0],
         [R10, R11, R12, t1],
         [R20, R21, R22, t2],
         [  0,   0,   0,  1]]
    where:
        - R is a 3x3 rotation matrix
        - t is a 3x1 translation vector
        - Bottom row must be [0, 0, 0, 1]

    The pose matrix is the inverse of the extrinsic matrix T.

    Args:
        pose (Float[np.ndarray, "4 4"]): Camera pose matrix to validate.

    Raises:
        ValueError: If pose is not a 4x4 matrix or bottom row is not [0, 0, 0, 1].
    """
    if pose.shape != (4, 4):
        raise ValueError(
            f"pose must has shape (4, 4), but got {pose} of shape {pose.shape}."
        )
    is_valid = np.allclose(pose[3, :], np.array([0, 0, 0, 1]))
    if not is_valid:
        raise ValueError(f"pose must has [0, 0, 0, 1] the bottom row, but got {pose}.")


def assert_shape(x: np.ndarray, shape: tuple, name: Optional[str] = None):
    """
    Assert that an array has the expected shape.

    The shape pattern can contain None values to indicate that dimension can be
    any size. For example:
        - (None, 3) matches any 2D array where the second dimension is 3
        - (3, None, 3) matches any 3D array where first and last dimensions are 3

    Args:
        x (np.ndarray): Array to validate.
        shape (tuple): Tuple of expected dimensions (can contain None for flexible
            dimensions).
        name (Optional[str]): Optional name of the variable for error message.
            Default: None.

    Raises:
        ValueError: If array dimensions don't match the expected shape pattern.
    """
    shape_valid = True

    if shape_valid and x.ndim != len(shape):
        shape_valid = False

    if shape_valid:
        for i, s in enumerate(shape):
            if s is not None:
                if x.shape[i] != s:
                    shape_valid = False
                    break

    if not shape_valid:
        name_must = f"{name} must" if name is not None else "Must"
        raise ValueError(f"{name_must} has shape {shape}, but got shape {x.shape}.")


def assert_shape_ndim(x: np.ndarray, ndim: int, name: Optional[str] = None):
    """
    Assert that x has exactly ndim dimensions.

    Args:
        x: Array to validate
        ndim: Expected number of dimensions
        name: Optional name of the variable for error message

    Raises:
        ValueError: If array doesn't have exactly ndim dimensions
    """
    if x.ndim != ndim:
        name_must = f"{name} must" if name is not None else "Must"
        raise ValueError(f"{name_must} have {ndim} dimensions, but got {x.ndim}.")


def assert_shape_nx3(x: np.ndarray, name: Optional[str] = None):
    """
    Assert that x is a 2D array with shape (N, 3).

    This is commonly used for arrays of 3D points or vectors.

    Args:
        x: Array to validate
        name: Optional name of the variable for error message
    """
    assert_shape(x, (None, 3), name=name)


def assert_shape_nx2(x: np.ndarray, name: Optional[str] = None):
    """
    Assert that x is a 2D array with shape (N, 2).

    This is commonly used for arrays of 2D points or vectors.

    Args:
        x: Array to validate
        name: Optional name of the variable for error message
    """
    assert_shape(x, (None, 2), name=name)


def assert_shape_4x4(x: np.ndarray, name: Optional[str] = None):
    """
    Assert that x is a 4x4 matrix.

    This is commonly used for transformation matrices like T or pose.

    Args:
        x: Array to validate
        name: Optional name of the variable for error message
    """
    assert_shape(x, (4, 4), name=name)


def assert_shape_3x4(x: np.ndarray, name: Optional[str] = None):
    """
    Assert that x is a 3x4 matrix.

    This is commonly used for camera projection matrices.

    Args:
        x: Array to validate
        name: Optional name of the variable for error message
    """
    assert_shape(x, (3, 4), name=name)


def assert_shape_3x3(x: np.ndarray, name: Optional[str] = None):
    """
    Assert that x is a 3x3 matrix.

    This is commonly used for rotation matrices or intrinsic matrices.

    Args:
        x: Array to validate
        name: Optional name of the variable for error message
    """
    assert_shape(x, (3, 3), name=name)


def assert_shape_3(x: np.ndarray, name: Optional[str] = None):
    """
    Assert that x is a 1D array with 3 elements.

    This is commonly used for 3D vectors or points.

    Args:
        x: Array to validate
        name: Optional name of the variable for error message
    """
    assert_shape(x, (3,), name=name)
