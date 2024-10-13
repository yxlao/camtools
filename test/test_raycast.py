import numpy as np
import open3d as o3d

import camtools as ct


def test_mesh_to_depth():
    # Geometries
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere = sphere.translate([0, 0, 4])
    box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=1.5)
    box = box.translate([0, 0, 4])
    mesh = sphere + box
    lineset = ct.convert.mesh_to_lineset(mesh)

    # Camera
    width, height = 640, 480
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T = np.eye(4)
    camera_frustum = ct.camera.create_camera_frustums([K], [T], size=1)

    # 3D -> 2D -> 3D projection (mesh -> depth -> points)
    im_depth = ct.raycast.mesh_to_im_depth(mesh, K, T, height, width)
    points = ct.project.im_depth_to_point_cloud(im_depth, K, T)

    # Verify points are on the mesh surface
    points_to_mesh_distances = ct.solver.points_to_mesh_distances(points, mesh)
    assert np.max(points_to_mesh_distances) < 5e-3

    visualize = False
    if visualize:
        import matplotlib.pyplot as plt

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([pcd, lineset, camera_frustum, axes])

        plt.figure(figsize=(10, 7.5))
        plt.imshow(im_depth, cmap="viridis")
        plt.colorbar(label="Depth")
        plt.title("Depth Image")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.show()
