import copy
from typing import List, Tuple

import numpy as np
import open3d as o3d

from . import convert, sanity


def render_geometries(
    geometries: List[o3d.geometry.Geometry3D],
    K: np.ndarray = None,
    T: np.ndarray = None,
    view_status_str: str = None,
    height: int = 720,
    width: int = 1280,
    visible: bool = False,
    point_size: float = 1.0,
) -> None:
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


def render_geometries_with_reset_light(
    geometries: List[o3d.geometry.Geometry3D],
    K: np.ndarray = None,
    T: np.ndarray = None,
    height: int = 720,
    width: int = 1280,
    visible: bool = False,
    point_size: float = 1.0,
):
    """
    Renders geometries with a transformed view by applying the inverse
    transformation to the geometries instead of setting a new camera view.

    Args:
        geometries: List of Open3D geometry objects to render.
        K: Camera intrinsic matrix. If None, Open3D's default is used.
        T: The target camera extrinsic matrix to simulate.
        height: The height of the rendered image.
        width: The width of the rendered image.
        visible: Whether the Open3D window is visible during rendering.
        point_size: Point size for point cloud objects.
    """
    K_default, T_default = get_render_K_T(
        geometries,
        K=K,
        T=None,
        height=height,
        width=width,
    )

    # Compute T compensation, where:
    # - render(geometries, T_new)
    # - render(geometries.transform(T_comp), T_default)
    # are equivalent.
    T_comp = np.linalg.inv(T_default) @ T

    geometries_transformed = [
        copy.deepcopy(geometry).transform(T_comp) for geometry in geometries
    ]

    return render_geometries(
        geometries_transformed,
        K=K,
        T=T_default,
        height=height,
        width=width,
        visible=visible,
        point_size=point_size,
    )


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
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = o3d_intrinsic
        cam_params.extrinsic = o3d_extrinsic
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(
            cam_params,
            allow_arbitrary=True,
        )

    vis.poll_events()
    vis.update_renderer()
    view_status_str = vis.get_view_status()
    vis.destroy_window()

    return view_status_str


def get_render_K_T(
    geometries: List[o3d.geometry.Geometry3D],
    K: np.ndarray = None,
    T: np.ndarray = None,
    view_status_str: str = None,
    height: int = 720,
    width: int = 1280,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the rendering camera intrinsic (K) and extrinsic (T) matrices set by Open3D.

    Args:
        geometries: List of Open3D geometries.
        K: (3, 3) np.ndarray camera intrinsic. If None, use Open3D's camera
            inferred from the geometries. You may optionally provide K or T.
        T: (4, 4) np.ndarray camera extrinsic. If None, use Open3D's camera
            inferred from the geometries. You may optionally provide K or T.
        view_status_str: Optional. The json string returned by
            o3d.visualization.Visualizer.get_view_status(), containing
            the viewing camera parameters. view_status_str takes precedence
            over K and T.
        height: int, image height.
        width: int, image width.

    Returns:
        K: (3, 3) np.ndarray camera intrinsic matrix.
        T: (4, 4) np.ndarray camera extrinsic matrix.
    """
    if not isinstance(geometries, list):
        raise TypeError("geometries must be a list of Open3D geometries.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=width,
        height=height,
        visible=False,
    )

    for geometry in geometries:
        vis.add_geometry(geometry)

    if view_status_str is not None:
        vis.set_view_status(view_status_str)

    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()

    # Use K or T if provided, but view_status_str takes precedence.
    if not view_status_str:
        if K is not None:
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
            )
            cam_params.intrinsic = o3d_intrinsic
        if T is not None:
            cam_params.extrinsic = T
        ctr.convert_from_pinhole_camera_parameters(
            cam_params,
            allow_arbitrary=True,
        )

        vis.poll_events()
        vis.update_renderer()
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()

    K = np.copy(np.array(cam_params.intrinsic.intrinsic_matrix))
    T = np.copy(np.array(cam_params.extrinsic))

    vis.destroy_window()

    return K, T
