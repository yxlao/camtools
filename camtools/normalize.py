import numpy as np


def compute_normalize_mat(points):
    """
    Args:
        points: (N, 3) numpy array.

    Returns:
        Returns normalize_mat, where `normalize_mat @ points_homo` is centered
        at the origin and is scaled within the unit sphere (max norm equals 1).

        You can check the correctness of compute_normalize_mat by:
        ```python
        normalize_mat = ct.normalize.compute_normalize_mat(points)
        points_normalized = ct.project.homo_project(points, normalize_mat)
        ct.stat.report_points_range(points_normalized)
        ```

        Typically, we also scale the camera after normalizing points. Given
        the camera parameter `K` and `T`, we can calculate `K_new` and `T_new`:
        ```python
        K_new = K
        C = ct.convert.T_to_C(T)
        C_new = ct.project.homo_project(C.reshape((-1, 3)), normalize_mat).flatten()
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
