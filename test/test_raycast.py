import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import camtools as ct


def distance_to_z_depth(distance_depth, K):
    """
    Convert distance depth to z-depth.

    Args:
        distance_depth (np.ndarray): Distance depth image.
        K (np.ndarray): Camera intrinsic matrix.

    Returns:
        np.ndarray: Z-depth image.
    """
    # Create a mask for valid depth values
    valid_mask = distance_depth > 0

    # Initialize z_depth with the same shape as distance_depth
    z_depth = np.zeros_like(distance_depth)

    # Extract focal lengths and principal points from K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Calculate pixel coordinates
    height, width = distance_depth.shape
    y, x = np.mgrid[0:height, 0:width]

    # Calculate normalized pixel coordinates
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy

    # Calculate z-depth for valid pixels
    z_depth[valid_mask] = distance_depth[valid_mask] / np.sqrt(
        1 + x_norm[valid_mask] ** 2 + y_norm[valid_mask] ** 2
    )

    return z_depth


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
    camera_frustum = ct.camera.create_camera_frustums([K], [T], size=2)

    # mesh -> depth
    im_distance = ct.raycast.mesh_to_distance(mesh, K, T, height, width)

    # Convert distance depth to z-depth
    im_depth = distance_to_z_depth(im_distance, K)

    # z-depth -> points
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
