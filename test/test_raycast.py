import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import camtools as ct
from codetiming import Timer


def mesh_to_lineset(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
    # Extract vertices and triangles from the mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Create edges from triangles
    edges = set()
    for triangle in triangles:
        edges.add(tuple(sorted([triangle[0], triangle[1]])))
        edges.add(tuple(sorted([triangle[1], triangle[2]])))
        edges.add(tuple(sorted([triangle[2], triangle[0]])))

    # Convert edges to a numpy array
    edges = np.array(list(edges))

    # Create a LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    return line_set


@Timer(name="distance_to_depth_v1")
def distance_to_depth_v1(im_distance, K):
    """
    Convert distance depth to z-depth.

    Args:
        im_distance (np.ndarray): Distance image.
        K (np.ndarray): Camera intrinsic matrix.

    Returns:
        np.ndarray: Z-depth image.
    """
    # Extract focal lengths and principal points from K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Calculate pixel coordinates
    height, width = im_distance.shape
    y, x = np.mgrid[0:height, 0:width]

    # Calculate normalized pixel coordinates
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy

    # Calculate z-depth
    z_depth = im_distance / np.sqrt(1 + x_norm**2 + y_norm**2)

    return z_depth


@Timer(name="distance_to_depth_v2")
def distance_to_depth_v2(im_distance, K):
    height, width = im_distance.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)

    u_norm = (u_grid - cx) / fx
    v_norm = (v_grid - cy) / fy
    norm_square = u_norm**2 + v_norm**2
    z_depth = im_distance / np.sqrt(norm_square + 1)

    return z_depth


@Timer(name="distance_to_depth_v3")
def distance_to_depth_v3(im_distance, K):
    """
    from marigold
    """
    # Extract focal length from K (assuming fx = fy)
    flt_focal = K[0, 0]

    # Get height and width from the distance array
    height, width = im_distance.shape

    img_plane_x = (
        np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width)
        .reshape(1, width)
        .repeat(height, 0)
        .astype(np.float32)[:, :, None]
    )
    img_plane_y = (
        np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height)
        .reshape(height, 1)
        .repeat(width, 1)
        .astype(np.float32)[:, :, None]
    )
    img_plane_z = np.full([height, width, 1], flt_focal, np.float32)
    img_plane = np.concatenate([img_plane_x, img_plane_y, img_plane_z], 2)

    depth = im_distance / np.linalg.norm(img_plane, 2, 2) * flt_focal
    return depth


def compute_point_to_mesh_distance(points, mesh):
    """
    Compute the distance from points to a mesh surface.

    Args:
        points (np.ndarray): Array of points with shape (N, 3).
        mesh (o3d.geometry.TriangleMesh): The input mesh.

    Returns:
        np.ndarray: Array of distances with shape (N,).
    """
    # Convert the legacy mesh to o3d.t.geometry.TriangleMesh
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a RaycastingScene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    # Convert points to o3d.core.Tensor
    points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    # Compute the unsigned distance from the points to the mesh surface
    distances = scene.compute_distance(points_tensor)

    # Convert distances to numpy array
    return distances.numpy()


def distance_to_depth(im_distance, K):
    height, width = im_distance.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)

    u_norm = (u_grid - cx) / fx
    v_norm = (v_grid - cy) / fy
    norm_square = u_norm**2 + v_norm**2
    z_depth = im_distance / np.sqrt(norm_square + 1)

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
    lineset = mesh_to_lineset(mesh)

    # Camera K and T
    width, height = 640, 480
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T = np.eye(4)
    camera_frustum = ct.camera.create_camera_frustums([K], [T], size=2)

    # mesh -> depth
    im_distance = ct.raycast.mesh_to_distance(mesh, K, T, height, width)

    # Convert distance to depth
    im_depth = distance_to_depth(im_distance, K).astype(np.float32)

    # z-depth -> points
    points = ct.project.depth_to_point_cloud(im_depth, K, T)

    # Compute distances
    distances = compute_point_to_mesh_distance(points, mesh)
    print(f"distances: max {np.max(distances)}, avg {np.mean(distances)}")

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([pcd, lineset, camera_frustum, axes])

    # # Plot the depth image (optional, you can keep or remove this part)
    # plt.figure(figsize=(10, 7.5))
    # plt.imshow(im_depth, cmap="viridis")
    # plt.colorbar(label="Depth")
    # plt.title("Depth Image")
    # plt.xlabel("Pixel X")
    # plt.ylabel("Pixel Y")
    # plt.show()


# Run the test function if this script is executed directly
if __name__ == "__main__":
    test_mesh_to_depth()
