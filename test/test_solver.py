import numpy as np
import camtools as ct

np.set_printoptions(formatter={"float": "{: 0.2f}".format})


def test_point_plane_distance_three_points():
    point = np.array([0, 0, 0])

    plane_points = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
    ref_dist = 1.0
    dist = ct.solver.point_plane_distance_three_points(point, plane_points)
    np.testing.assert_allclose(dist, ref_dist)

    plane_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ref_dist = 1 / np.sqrt(3)
    dist = ct.solver.point_plane_distance_three_points(point, plane_points)
