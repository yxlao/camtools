import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import camtools as ct


def test_mesh_to_depth():
    # Geometries
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=1.5)
    sphere.translate([0, 0, 2])
    box.translate([0, 0, 2])
    sphere.compute_vertex_normals()
    box.compute_vertex_normals()
    mesh = sphere + box

    # Camera K and T
    width, height = 640, 480
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T = np.eye(4)
    camera_frustum = ct.camera.create_camera_frustums([K], [T], size=0.5)

    # mesh -> depth
    im_depth = ct.raycast.mesh_to_depth(mesh, K, T, height, width)

    # depth -> points
    points = ct.project.depth_to_point_cloud(im_depth, K, T)

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([pcd, mesh, camera_frustum, axes])

    # # Plot the depth image (optional, you can keep or remove this part)
    # plt.figure(figsize=(10, 7.5))
    # plt.imshow(im_depth, cmap="viridis")
    # plt.colorbar(label="Depth")
    # plt.title("Depth Image")
    # plt.xlabel("Pixel X")
    # plt.ylabel("Pixel Y")
    # plt.show()
