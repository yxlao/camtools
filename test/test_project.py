import numpy as np
import open3d as o3d

import camtools as ct


def test_points_to_depths(visualize=True):
    # Identity camera pose (looking at +Z axis)
    T = np.eye(4)
    fx = fy = 500
    cx = 320
    cy = 240
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Create points on a plane z units away from origin
    num_points = 5000
    z = 2.0
    x = np.random.uniform(-2, 2, num_points)
    y = np.random.uniform(-1.5, 1.5, num_points)
    points = np.column_stack([x, y, np.full(num_points, z)])

    # Test depths are all close to z
    depths = ct.project.points_to_depths(points, T)
    assert np.allclose(
        depths, z
    ), f"Depths should be {z}, got {depths.min():.2f}-{depths.max():.2f}"

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1, 0, 0])
        frustum = ct.camera.create_camera_frustums(
            Ks=[K],
            Ts=[T],
            size=1.0,
        )
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([pcd, frustum, axes])
