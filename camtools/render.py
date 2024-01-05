from typing import List

import numpy as np
import open3d as o3d

from . import sanity


def render_geometries(
    geometries: List[o3d.geometry.Geometry3D],
    K: np.ndarray = None,
    T: np.ndarray = None,
    view_status_str: str = None,
    height: int = 720,
    width: int = 1280,
    visible: bool = False,
    point_size: float = 1.0,
):
    """
    Render Open3D geometries to an image. This function may require a display.

    Args:
        mesh: Open3d TriangleMesh.
        K: (3, 3) np.ndarray camera intrinsic. If None, use Open3D's camera
            inferred from the geometries. K must be provided if T is provided.
        T: (4, 4) np.ndarray camera extrinsic. If None, use Open3D's camera
            inferred from the geometries. T must be provided if K is provided.
        view_status_str: The json string returned by
            o3d.visualization.Visualizer.get_view_status(), containing
            the viewing camera parameters. This does not include the window
            size and the point size.
        height: int image height.
        width: int image width.
        visible: bool whether to show the window.
        point_size: float point size for point cloud objects.

    Returns:
        image: (H, W, 3) float32 np.ndarray image.
    """
    if not isinstance(geometries, list):
        raise TypeError("geometries must be a list of Open3D geometries.")
    if K is None and T is not None:
        raise ValueError("K must be provided if T is provided.")
    elif K is not None and T is None:
        raise ValueError("T must be provided if K is provided.")
    elif K is None and T is None:
        is_camera_provided = False
    else:
        is_camera_provided = True
        sanity.assert_K(K)
        sanity.assert_T(T)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=width,
        height=height,
        visible=visible,
    )

    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.PointCloud):
            vis.get_render_option().point_size = point_size
        vis.add_geometry(geometry)

    if is_camera_provided:
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )
        o3d_extrinsic = T
        o3d_camera = o3d.camera.PinholeCameraParameters()
        o3d_camera.intrinsic = o3d_intrinsic
        o3d_camera.extrinsic = o3d_extrinsic
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(
            o3d_camera,
            allow_arbitrary=True,
        )
        for geometry in geometries:
            vis.update_geometry(geometry)

    if view_status_str is not None:
        vis.set_view_status(view_status_str)

    vis.poll_events()
    vis.update_renderer()
    buffer = vis.capture_screen_float_buffer()
    vis.destroy_window()
    im_buffer = np.asarray(buffer)

    return im_buffer


def get_render_view_status_str(
    geometries: List[o3d.geometry.Geometry3D],
    K: np.ndarray = None,
    T: np.ndarray = None,
    height: int = 720,
    width: int = 1280,
) -> str:
    """
    Get a view status string for rendering with Open3D visualizer. This is
    useful for rendering multiple geometries with the same rendering camera.
    This function may require a display.

    Args:
        geometries: List of Open3D geometries.
        K: (3, 3) np.ndarray camera intrinsic. If None, use Open3D's camera
            inferred from the geometries. K must be provided if T is provided.
        T: (4, 4) np.ndarray camera extrinsic. If None, use Open3D's camera
            inferred from the geometries. T must be provided if K is provided.
        height: int image height.
        width: int image width.

    Returns:
        view_status_str: The json string returned by
            o3d.visualization.Visualizer.get_view_status(), containing
            the viewing camera parameters. This does not include the window
            size and the point size.
    """
    if not isinstance(geometries, list):
        raise TypeError("geometries must be a list of Open3D geometries.")
    if K is None and T is not None:
        raise ValueError("K must be provided if T is provided.")
    elif K is not None and T is None:
        raise ValueError("T must be provided if K is provided.")
    elif K is None and T is None:
        is_camera_provided = False
    else:
        is_camera_provided = True
        sanity.assert_K(K)
        sanity.assert_T(T)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=width,
        height=height,
        visible=False,
    )

    for geometry in geometries:
        vis.add_geometry(geometry)

    if is_camera_provided:
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )
        o3d_extrinsic = T
        o3d_camera = o3d.camera.PinholeCameraParameters()
        o3d_camera.intrinsic = o3d_intrinsic
        o3d_camera.extrinsic = o3d_extrinsic
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(
            o3d_camera,
            allow_arbitrary=True,
        )

    vis.poll_events()
    vis.update_renderer()
    view_status_str = vis.get_view_status()
    vis.destroy_window()

    return view_status_str
