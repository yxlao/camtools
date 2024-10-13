import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import camtools as ct


def test_mesh_to_depth():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=1.5)
    sphere.compute_vertex_normals()
    box.compute_vertex_normals()

    # Move the geometries
    sphere.translate([0, 0, 4])
    box.translate([0, 0, 4])

    # Combine the geometries
    mesh = sphere + box

    # Create dummy camera intrinsics (K)
    width, height = 640, 480
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T = np.eye(4)

    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # camera_frustum = ct.camera.create_camera_frustums([K], [T], size=1)
    # o3d.visualization.draw_geometries([camera_frustum, mesh, axes])

    # Call mesh_to_depth function
    im_depth = ct.raycast.mesh_to_depth(mesh, K, T, height, width)

    # Convert depth image to point cloud
    points = ct.project.depth_to_point_cloud(im_depth, K, T)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud along with the original meshes
    o3d.visualization.draw_geometries([pcd, mesh])

    # Plot the depth image (optional, you can keep or remove this part)
    plt.figure(figsize=(10, 7.5))
    plt.imshow(im_depth, cmap="viridis")
    plt.colorbar(label="Depth")
    plt.title("Depth Image")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()
