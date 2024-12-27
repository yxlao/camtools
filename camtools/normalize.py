import numpy as np
from typing import Tuple
from jaxtyping import Float


def compute_normalize_mat(points: Float[np.ndarray, "n 3"]) -> Float[np.ndarray, "4 4"]:
    """
    Args:
        points: (N, 3) numpy array.

    Returns:
        Returns normalize_mat, where `normalize_mat @ points_homo` is centered
        at the origin and is scaled within the unit sphere (max norm equals 1).

        You can check the correctness of compute_normalize_mat by:
        ```python
        normalize_mat = ct.normalize.compute_normalize_mat(points)
        points_normalized = ct.transform.transform_points(points, normalize_mat)
        ct.stat.report_points_range(points_normalized)
        ```

        Typically, we also scale the camera after normalizing points. Given
        the camera parameter `K` and `T`, we can calculate `K_new` and `T_new`:
        ```python
        K_new = K
        C = ct.convert.T_to_C(T)
        C_new = ct.transform.transform_points(C.reshape((-1, 3)), normalize_mat).flatten()
        pose_new = np.linalg.inv(T)
        pose_new[:3, 3] = C_new
        T_new = np.linalg.inv(pose_new)
        ```
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), but got {points.shape}.")

    # Translate
    points_center = points.mean(axis=0)
    t = -points_center
    t_mat = np.eye(4)
    t_mat[:3, 3] = t

    # Scale
    max_norm = np.linalg.norm(points - points_center, axis=1).max()
    s = 1.0 / max_norm
    s_mat = np.eye(4) * s
    s_mat[3, 3] = 1

    return s_mat @ t_mat


def report_points_range(points: Float[np.ndarray, "n 3"]) -> None:
    """
    Report statistics about the points.

    Args:
        points: (N, 3) numpy array.

    Prints:
        - Center of the points
        - Maximum radius with respect to center
        - Maximum radius with respect to origin
        - Range of points (min and max coordinates)
    """
    points_center = points.mean(axis=0)
    points_radii_wrt_center = np.linalg.norm(points - points_center, axis=1)
    points_radii_wrt_origin = np.linalg.norm(points, axis=1)

    print(f"center             : {points_center}")
    print(f"radius w.r.t center: {points_radii_wrt_center.max()}")
    print(f"radius w.r.t origin: {points_radii_wrt_origin.max()}")
    print(f"range              : {points.min(axis=0)} to {points.max(axis=0)}")
