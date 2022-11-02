import numpy as np


def report_points_range(points):
    points_center = points.mean(axis=0)
    points_radii_wrt_center = np.linalg.norm(points - points_center, axis=1)
    points_radii_wrt_origin = np.linalg.norm(points, axis=1)

    print(f"center             : {points_center}")
    print(f"radius w.r.t center: {points_radii_wrt_center.max()}")
    print(f"radius w.r.t origin: {points_radii_wrt_origin.max()}")
    print(f"range              : {points.min(axis=0)} to {points.max(axis=0)}")
