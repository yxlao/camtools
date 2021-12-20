import numpy as np


def compute_normalize_mat(points):
    """
    Args:
        points: (N, 3) numpy array.

    Returns:
        Returns normalize_mat, where `normalize_mat @ points_homo` is centered
        at the origin and is scaled within the unit sphere (max norm equals 1).

        In NeuS setting, normalize_mat == scale_mat.inv(). The point is
        projected by:
        pixels_x_y_1 = world_mat @ scale_mat @ normalize_mat @ points_X_Y_Z_1

        You can check the correctness of compute_normalize_mat by

        ```python
        normalize_mat = ct.normalize.compute_normalize_mat(points)
        points_normalized = ct.project.homo_project(normalize_mat, points)
        ct.stat.report_points_range(points_normalized)
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
