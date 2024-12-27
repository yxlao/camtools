from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from jaxtyping import Float

from . import artifact, image, sanity


def render_geometries(
    geometries: List[o3d.geometry.Geometry3D],
    K: Optional[Float[np.ndarray, "3 3"]] = None,
    T: Optional[Float[np.ndarray, "4 4"]] = None,
    view_status_str: Optional[str] = None,
    height: int = 720,
    width: int = 1280,
    point_size: float = 1.0,
    line_radius: Optional[float] = None,
    to_depth: bool = False,
    visible: bool = False,
) -> Float[np.ndarray, "h w 3"]:
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
        point_size: float point size for point cloud objects.
        line_radius: float line radius for line set objects, when set, the line
            sets will be converted to cylinder meshes with the given radius.
            The radius is in world metric space, not relative pixel space like
            the point size.
        to_depth: bool whether to render a depth image instead of RGB image.
            Invalid depth or infinite depth is set to 0.
        visible: bool whether to show the window.

    Returns:
        (H, W, 3) float32 np.ndarray RGB image by default; (H, W) float32
        depth image if to_depth is True.
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
    if to_depth:
        buffer = vis.capture_depth_float_buffer()
    else:
        buffer = vis.capture_screen_float_buffer()
    vis.destroy_window()
    im_buffer = np.asarray(buffer).astype(np.float32)

    return im_buffer


def get_render_view_status_str(
    geometries: List[o3d.geometry.Geometry3D],
    K: Optional[Float[np.ndarray, "3 3"]] = None,
    T: Optional[Float[np.ndarray, "4 4"]] = None,
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
    view_status_str: Optional[str] = None,
    height: int = 720,
    width: int = 1280,
) -> Tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "4 4"]]:
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


class _TextRenderer:
    """
    Renders text into an image using specified font settings.
    """

    FONT_MAP = {
        "tex": "a1/texgyrepagella-regular.otf",
        "serif": None,
        "sans": None,
        "mono": None,
    }

    def __init__(self, font_type: str = "tex"):
        """
        Initializes the renderer with a specific font type.
        """
        if font_type not in self.FONT_MAP:
            raise ValueError(
                f"Invalid font_type: {font_type}. "
                f"Available options: {list(self.FONT_MAP.keys())}."
            )
        artifact_key = self.FONT_MAP[font_type]
        if artifact_key is None:
            raise NotImplementedError(
                f"Font type '{font_type}' is not implemented yet."
            )

        self.font_path = artifact.get_artifact_path(artifact_key)

    def _get_text_size(
        self, text: str, font: ImageFont.FreeTypeFont, alignment: str
    ) -> Tuple[int, int, int, int]:
        """
        Estimates the full and tight sizes of the given text.

        Args:
            text: The text to measure.
            font: The font used for the text.

        Returns:
            A tuple containing the full width and height of the text box,
            as well as the tight width and height of the content within the
            text box.
        """
        im = Image.new(mode="RGB", size=(1, 1))
        draw = ImageDraw.Draw(im)
        bbox = draw.textbbox((0, 0), text=text, font=font, align=alignment)

        full_w, full_h = bbox[2], bbox[3]
        tight_w, tight_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Well, they shall be integers, but can be something like 32.0
        full_w = int(round(full_w))
        full_h = int(round(full_h))
        tight_w = int(round(tight_w))
        tight_h = int(round(tight_h))

        return full_w, full_h, tight_w, tight_h

    def render(
        self,
        text: str,
        font_size: int,
        font_color: Tuple[float, float, float],
        tight_layout: bool,
        multiline_alignment: str,
    ) -> np.ndarray:
        """
        Renders the given text with specified settings.

        Args:
            text: The text to render.
            font_size: The font size to use.
            font_color: The color of the font, as an RGB tuple in the
                range [0, 1].
            tight_layout: If True, renders the text without any padding
                around it. If False, may include some padding on top, aligning
                letters by the top for consistent alignment across images.
            alignment: The alignment of the text. Can be "left", "center",
                or "right", this is useful for multi-line text.

        Returns:
            The rendered text as a NumPy array (float32).
        """
        # Sanity checks
        if len(font_color) != 3 or not all(0 <= c <= 1 for c in font_color):
            raise ValueError(
                f"font_color must be 3 floats in the range [0, 1], "
                f"but got {font_color}."
            )
        if multiline_alignment not in ["left", "center", "right"]:
            raise ValueError(
                f"Invalid alignment: {multiline_alignment}, must be left, center, or right."
            )

        # Init font
        font = ImageFont.truetype(str(self.font_path), size=font_size)

        # Compute dimensions
        sizes = self._get_text_size(text, font, multiline_alignment)
        full_w, full_h, tight_w, tight_h = sizes
        w_gap = full_w - tight_w
        h_gap = full_h - tight_h
        if tight_layout:
            im_w = tight_w
            im_h = tight_h
            pos = (-w_gap, -h_gap)
        else:
            im_w = full_w
            im_h = full_h
            pos = (0, 0)

        # Render
        im_render = Image.new("RGB", (im_w, im_h), "white")
        draw = ImageDraw.Draw(im_render)
        color_uint8 = tuple(int(c * 255) for c in font_color)
        draw.multiline_text(
            pos,
            text,
            fill=color_uint8,
            font=font,
            align=multiline_alignment,
        )
        im_render = np.asarray(im_render).astype(np.float32) / 255.0

        return im_render


def render_text(
    text: str,
    font_size: int = 72,
    font_type: str = "tex",
    font_color: Tuple[float, float, float] = (0, 0, 0),
    tight_layout: bool = False,
    multiline_alignment: str = "left",
    padding_tblr: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """
    Global function to render text using specified font settings.

    Args:
        text: The text to render.
        font_size: The font size to use.
        font_type: The type of font.
        font_color: The color of the font, as an RGB tuple in the
            range [0, 1].
        tight_layout: If True, renders the text without padding. If False,
            may include padding on top for top alignment in images.
        alignment: The alignment of the text. Can be "left", "center",
            or "right", this is useful for multi-line text.
        padding_tblr: The padding to add to the top, bottom, left, and right
            of the rendered text, in pixels.

    Returns:
        The rendered text as a NumPy array (float32).
    """
    if (
        len(padding_tblr) != 4
        or not all(p >= 0 for p in padding_tblr)
        or not all(isinstance(p, int) for p in padding_tblr)
    ):
        raise ValueError(
            f"padding_tblr must be a tuple of 4 non-negative integers, "
            f"but got {padding_tblr}."
        )

    im_render = _TextRenderer(font_type=font_type).render(
        text=text,
        font_size=font_size,
        font_color=font_color,
        tight_layout=tight_layout,
        multiline_alignment=multiline_alignment,
    )

    if padding_tblr != (0, 0, 0, 0):
        im_render = np.pad(
            im_render,
            (
                (padding_tblr[0], padding_tblr[1]),
                (padding_tblr[2], padding_tblr[3]),
                (0, 0),
            ),
            mode="constant",
            constant_values=1.0,
        )

    return im_render


def render_texts(
    texts: List[str],
    font_size: int = 72,
    font_type: str = "tex",
    font_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    multiline_alignment: str = "center",
    same_height: bool = False,
    same_width: bool = False,
    padding_tblr: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> List[np.ndarray]:
    if (
        len(padding_tblr) != 4
        or not all(p >= 0 for p in padding_tblr)
        or not all(isinstance(p, int) for p in padding_tblr)
    ):
        raise ValueError(
            f"padding_tblr must be a tuple of 4 non-negative integers, "
            f"but got {padding_tblr}."
        )

    im_renders = [
        render_text(
            text,
            font_size=font_size,
            font_type=font_type,
            font_color=font_color,
            tight_layout=False,
            multiline_alignment=multiline_alignment,
        )
        for text in texts
    ]

    if same_height:
        max_height = max(im.shape[0] for im in im_renders)
        im_renders = [
            np.pad(
                im,
                ((0, max_height - im.shape[0]), (0, 0), (0, 0)),
                mode="constant",
                constant_values=1.0,
            )
            for im in im_renders
        ]

    if same_width:
        max_width = max(im.shape[1] for im in im_renders)
        im_renders = [
            np.pad(
                im,
                (
                    (0, 0),
                    (
                        (max_width - im.shape[1]) // 2,
                        max_width - im.shape[1] - (max_width - im.shape[1]) // 2,
                    ),
                    (0, 0),
                ),
                mode="constant",
                constant_values=1.0,
            )
            for im in im_renders
        ]

    if padding_tblr != (0, 0, 0, 0):
        im_renders = [
            np.pad(
                im,
                (
                    (padding_tblr[0], padding_tblr[1]),
                    (padding_tblr[2], padding_tblr[3]),
                    (0, 0),
                ),
                mode="constant",
                constant_values=1.0,
            )
            for im in im_renders
        ]

    return im_renders
