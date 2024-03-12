from typing import List, Tuple

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
    line_radius: float = None,
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
        line_radius: float line radius for line set objects, when set, the line
            sets will be converted to cylinder meshes with the given radius.
            The radius is in world metric space, not relative pixel space like
            the point size.

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

    if line_radius is not None:
        geometries = _preprocess_geometries_lineset_to_meshes(
            geometries=geometries, line_radius=line_radius
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


def get_render_K_T(
    geometries: List[o3d.geometry.Geometry3D],
    view_status_str: str = None,
    height: int = 720,
    width: int = 1280,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the rendering camera intrinsic (K) and extrinsic (T) matrices set by Open3D.

    Args:
        geometries: List of Open3D geometries.
        view_status_str: Optional. The json string returned by
            o3d.visualization.Visualizer.get_view_status(), containing
            the viewing camera parameters.
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

    K = np.copy(np.array(cam_params.intrinsic.intrinsic_matrix))
    T = np.copy(np.array(cam_params.extrinsic))

    vis.destroy_window()

    return K, T


def _preprocess_geometries_lineset_to_meshes(
    geometries: List[o3d.geometry.Geometry3D],
    line_radius: float,
) -> List[o3d.geometry.Geometry3D]:
    """
    Preprocess geometries by converting LineSet objects to TriangleMeshes.
    All other geometries are left unchanged.
    """
    new_geometries = []
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.LineSet):
            new_geometries.extend(_lineset_to_meshes(geometry, line_radius))
        else:
            new_geometries.append(geometry)
    return new_geometries


def _lineset_to_meshes(
    line_set: o3d.geometry.LineSet,
    radius: float,
) -> List[o3d.geometry.TriangleMesh]:
    """
    Converts an Open3D LineSet object to a list of mesh objects, preserving
    the line color and allowing the setting of line width.

    Args:
        line_set (o3d.geometry.LineSet): The line set to convert.
        radius (float): The radius (thickness) of the lines in the mesh. The
            unit is in actual metric space, not pixel space.

    Returns:
        List[o3d.geometry.TriangleMesh]: A list of TriangleMesh objects
        representing the lines.

    Reference:
        https://github.com/isl-org/Open3D/pull/738#issuecomment-564785941
        License: MIT
    """

    def align_vector_to_another(
        a: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        if np.allclose(a, b):
            return np.array([0, 0, 1]), 0.0
        axis = np.cross(a, b)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(
            np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1.0, 1.0)
        )
        return axis, angle

    def normalized(a: np.ndarray) -> Tuple[np.ndarray, float]:
        norm = np.linalg.norm(a)
        return (a / norm, norm) if norm != 0 else (a, 0.0)

    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)

    # Handle colors: default to black if no colors are provided
    if line_set.has_colors():
        colors = np.asarray(line_set.colors)
        if len(colors) != len(lines):
            raise ValueError("Number of colors must match number of lines.")
    else:
        colors = np.array([[0, 0, 0] for _ in range(len(lines))])

    cylinders = []
    for line, color in zip(lines, colors):
        start_point, end_point = points[line[0]], points[line[1]]
        line_segment = end_point - start_point
        line_segment_unit, line_length = normalized(line_segment)
        axis, angle = align_vector_to_another(np.array([0, 0, 1]), line_segment_unit)
        translation = start_point + line_segment * 0.5
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, line_length)
        cylinder.translate(translation, relative=False)
        if not np.isclose(angle, 0):
            axis_angle = axis * angle
            cylinder.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle),
                center=cylinder.get_center(),
            )
        cylinder.paint_uniform_color(color)
        cylinders.append(cylinder)

    return cylinders
