from pathlib import Path
import numpy as np
import open3d as o3d
import camtools as ct
import matplotlib.pyplot as plt


def test_render_geometries(visualize=True):
    # Setup geometries: sphere (red), box (blue)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=100)
    sphere = sphere.translate([0, 0, 4])
    sphere = sphere.paint_uniform_color([0.2, 0.4, 0.8])
    sphere.compute_vertex_normals()
    box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=1.5)
    box = box.translate([0, 0, 4])
    box = box.paint_uniform_color([0.8, 0.2, 0.2])
    box.compute_vertex_normals()
    mesh = sphere + box

    # Setup camera
    width, height = 640, 480
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T = np.eye(4)
    camera_frustum = ct.camera.create_camera_frustums([K], [T], size=1)

    # Test render
    im_render_rgb = ct.render.render_geometries(
        geometries=[mesh],
        K=K,
        T=T,
        height=height,
        width=width,
    )
    im_render_depth = ct.render.render_geometries(
        geometries=[mesh],
        K=K,
        T=T,
        height=height,
        width=width,
        to_depth=True,
    )
    im_raycast_depth = ct.raycast.mesh_to_im_depth(
        mesh=mesh,
        K=K,
        T=T,
        height=height,
        width=width,
    )

    # Heuristic checks of RGB rendering
    assert im_render_rgb.shape == (height, width, 3), "Image has incorrect dimensions"
    num_white_pixels = np.sum(
        (im_render_rgb[:, :, 0] > 0.9)
        & (im_render_rgb[:, :, 1] > 0.9)
        & (im_render_rgb[:, :, 2] > 0.9)
    )
    num_blue_pixels = np.sum(
        (im_render_rgb[:, :, 2] > 0.7)
        & (im_render_rgb[:, :, 0] < 0.3)
        & (im_render_rgb[:, :, 1] < 0.5)
    )
    num_red_pixels = np.sum(
        (im_render_rgb[:, :, 0] > 0.7)
        & (im_render_rgb[:, :, 1] < 0.3)
        & (im_render_rgb[:, :, 2] < 0.5)
    )
    assert num_white_pixels > (height * width * 0.5), "Expected mostly white background"
    assert num_blue_pixels > 100, "Expected blue pixels (sphere) not found"
    assert num_red_pixels > 100, "Expected red pixels (box) not found"

    # Create masks from RGB and depth
    im_rgb_mask = np.any(im_render_rgb < 0.99, axis=-1)
    im_raycast_mask = im_raycast_depth != np.inf  # Invalid depth is set to inf
    im_render_mask = im_render_depth > 0  # Invalid depth is set to 0

    # Compare masks
    im_mask_diff_raycast = np.abs(
        im_rgb_mask.astype(float) - im_raycast_mask.astype(float)
    )
    im_mask_diff_render = np.abs(
        im_rgb_mask.astype(float) - im_render_mask.astype(float)
    )
    im_mask_diff_methods = np.abs(
        im_raycast_mask.astype(float) - im_render_mask.astype(float)
    )
    assert (
        np.mean(im_mask_diff_raycast) < 0.01
    ), "RGB and raycast depth masks differ significantly"
    assert (
        np.mean(im_mask_diff_render) < 0.01
    ), "RGB and render depth masks differ significantly"
    assert (
        np.mean(im_mask_diff_methods) < 0.01
    ), "Raycast and render depth masks differ significantly"

    # Visualization
    if visualize:
        plt.figure(figsize=(25, 7.5))

        # RGB and depth images
        plt.subplot(1, 5, 1)
        plt.imshow(im_render_rgb)
        plt.title("Render RGB")

        plt.subplot(1, 5, 2)
        plt.imshow(im_render_depth, cmap="viridis")
        plt.colorbar(label="Depth")
        plt.title("Render Depth")

        plt.subplot(1, 5, 3)
        plt.imshow(im_raycast_depth, cmap="viridis")
        plt.colorbar(label="Depth")
        plt.title("Raycast Depth")

        # Mask comparisons
        plt.subplot(1, 5, 4)
        plt.imshow(im_mask_diff_render, cmap="gray")
        plt.title("RGB vs Render Depth Mask Diff")

        plt.subplot(1, 5, 5)
        plt.imshow(im_mask_diff_methods, cmap="gray")
        plt.title("Render vs Raycast Depth Mask Diff")

        plt.tight_layout()
        plt.show()

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([mesh, camera_frustum, axes])