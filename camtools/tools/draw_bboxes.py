import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Button

import camtools as ct
import argparse


@dataclass
class BBoxerConfig:
    """Configuration constants for BBoxer."""

    # Drawing properties
    default_linewidth: int = 1
    default_edgecolor: str = "red"

    # Mode defaults
    default_square_mode: bool = True
    default_enlarged_view_mode: bool = True

    # Enlarged view settings
    enlarged_view_scale_ratio: float = 0.75  # Fraction of shorter side (3/4)
    enlarged_view_min_ratio: float = 0.1  # Minimum size as fraction of max (10%)
    enlarged_view_step_ratio: float = 0.1  # Size adjustment step as fraction (10%)
    enlarged_view_margin: int = 10  # Pixels from edge when placing enlarged view

    # Dashed line settings
    dash_length: int = 10  # Length of each dash in pixels

    # Figure layout settings
    base_width_per_image: int = 8  # Inches per image
    max_fig_width: int = 24  # Maximum figure width in inches
    fig_height: int = 8  # Figure height in inches

    # Rectangle selector settings
    min_span_pixels: int = 5  # Minimum span for rectangle selector

    # Button layout settings
    button_height: float = 0.04
    button_small_width: float = 0.05  # For +/- buttons
    button_medium_width: float = 0.15  # For toggle buttons
    button_font_size: int = 8
    button_spacing: float = 0.01
    button_row1_y: float = 0.08  # Top row Y position
    button_row2_y: float = 0.02  # Bottom row Y position
    button_label_font_size: int = 9  # Font size for labels like "Line:" and "Zoom:"

    # Layout adjustments
    bottom_margin: float = 0.15  # Space reserved for buttons at bottom


class BBoxer:
    """
    Draw bounding boxes on images.
    """

    def __init__(self, linewidth=None, edgecolor=None, config=None):
        # Configuration.
        self.config = config if config is not None else BBoxerConfig()

        # Draw properties.
        self.linewidth = (
            linewidth if linewidth is not None else self.config.default_linewidth
        )
        self.edgecolor = (
            edgecolor if edgecolor is not None else self.config.default_edgecolor
        )

        # Input image paths.
        self.src_paths: List[Path] = []

        # Original images (never modified).
        self.original_images = []

        # Current rendered images (with bbox/enlarged view drawn).
        # These are the exact images shown in preview and saved to disk.
        self.current_rendered_images = []

        # Bounding box - only one at a time.
        self.current_bbox = None  # Tuple: (x0, y0, x1, y1) or None

        # Matplotlib objects.
        self.fig = None
        self.axes = []
        self.axis_images = []  # Store image objects for updating
        self.axis_to_selector = dict()
        self.button_linewidth_increase = None
        self.button_linewidth_decrease = None
        self.button_square_mode = None
        self.button_enlarged_view = None
        self.button_enlarged_size_increase = None
        self.button_enlarged_size_decrease = None

        # Mode states.
        self.square_mode = self.config.default_square_mode
        self.enlarged_view_mode = self.config.default_enlarged_view_mode
        self.enlarged_view_scale = None  # Will be set based on image size

    def add_paths(self, paths: List[Path]) -> None:
        """
        Append input image paths to list of images to be processed.
        The images must have the same dimensions and 3 channels.
        """
        for path in paths:
            self.src_paths.append(Path(path))

    def run(self) -> None:
        """
        Run the bounding boxer.

        Steps:
            1. Load images. Images must have the same dimensions and 3 channels.
            2. Display images simultaneously, side-by-side.
            3. Interactively draw bounding boxes on images.
            4. Press Enter to save and quit.
        """
        BBoxer._print_help_message()

        if len(self.src_paths) == 0:
            raise ValueError("No input images.")

        # Load and store original images.
        for src_path in self.src_paths:
            im_src = ct.io.imread(src_path)
            if im_src.ndim != 3 or im_src.shape[2] != 3:
                raise ValueError(f"Invalid image shape {im_src.shape}.")
            self.original_images.append(im_src)

        # Check all images are of the same shape.
        for im_src in self.original_images:
            if im_src.shape != self.original_images[0].shape:
                raise ValueError(
                    "Images must have the same shape "
                    f"{im_src.shape} != {self.original_images[0].shape}"
                )

        # Set default enlarged view scale based on config ratio.
        im_h, im_w = self.original_images[0].shape[:2]
        shorter_side = min(im_h, im_w)
        self.enlarged_view_scale = int(
            shorter_side * self.config.enlarged_view_scale_ratio
        )
        self.enlarged_view_scale_max = shorter_side

        # Initialize current rendered images (initially same as originals).
        self.current_rendered_images = list(self.original_images)

        # Register fig and axes with reasonable initial size.
        # Scale width based on number of images, but cap it to avoid excessive size.
        num_images = len(self.original_images)
        fig_width = min(
            self.config.base_width_per_image * num_images, self.config.max_fig_width
        )
        self.fig, self.axes = plt.subplots(
            1, num_images, figsize=(fig_width, self.config.fig_height)
        )
        if len(self.original_images) == 1:
            self.axes = [self.axes]

        # Display images and store image objects for later updates.
        for i, (axis, im_src) in enumerate(zip(self.axes, self.original_images)):
            img_obj = axis.imshow(im_src)
            self.axis_images.append(img_obj)
            axis.set_title(self.src_paths[i].name)
            axis.set_axis_off()
        plt.tight_layout()

        # Register rectangle selector callbacks.
        for axis in self.axes:
            self._register_rectangle_selector(axis)

        # Register other handlers.
        self.fig.canvas.mpl_connect("key_press_event", self._on_keypress)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)

        # Create all interactive buttons.
        self._create_buttons()

        plt.show()

    @staticmethod
    def _overlay_rectangle_on_image(
        im: np.ndarray,
        tl_xy: Tuple[int, int],
        br_xy: Tuple[int, int],
        linewidth_px: int,
        edgecolor: str,
        squarecorners: bool = True,
    ) -> np.ndarray:
        """
        Draw red rectangletangular bounding box on image using OpenCV.

        Args:
            im: Image to draw bounding box on. Must be float32.
            tl_xy: Top-left corner of bounding box, in (x, y), or (c, r).
                If the thickness is larger than 1, this is the center of the line.
            br_xy: Bottom-right corner of bounding box, in (x, y), or (c, r).
                If the thickness is larger than 1, this is the center of the line.
            linewidth: Width of bounding box line, this is in pixels!
            edgecolor: Color of bounding box line.
            squarecorners: If True, draw square corners. If False, draw rounded
                corners as opencv default.
        """

        def fill_connected_component(mat, x, y):
            """
            Mat: (h, w) single channel float32 image.
                0.0: empty pixel.
                1.0: filled pixel.
                -1.0: invalid pixel.
            x: x coordinate of pixel to start filling from.
            y: y coordinate of pixel to start filling from.
            """
            mat = np.copy(mat)
            # mat can only contain 0, -1, or 1.
            assert np.all(np.isin(mat, [-1.0, 0.0, 1.0]))
            # mat must be single channel.
            assert mat.ndim == 2
            # mat must be np.float32.
            assert mat.dtype == np.float32
            # Iterative DFS
            stack = [(x, y)]
            while len(stack) > 0:
                x, y = stack.pop()
                if mat[y, x] != 0.0:
                    continue
                mat[y, x] = 1.0
                if x > 0:
                    stack.append((x - 1, y))
                if x < mat.shape[1] - 1:
                    stack.append((x + 1, y))
                if y > 0:
                    stack.append((x, y - 1))
                if y < mat.shape[0] - 1:
                    stack.append((x, y + 1))
            return mat

        # Sanity checks.
        if im.dtype != np.float32:
            raise ValueError(f"Invalid image dtype {im.dtype}.")

        w = im.shape[1]
        h = im.shape[0]
        im_mask = np.zeros((h, w), dtype=np.float32)

        # Draw white lines on black im_mask. These lines have "rounded" corners.
        # Later we will fill "rounded" corners to square corners.
        cv2.rectangle(
            im_mask,
            pt1=tl_xy,
            pt2=br_xy,
            color=1.0,
            thickness=linewidth_px,
            lineType=cv2.LINE_8,
        )

        if squarecorners:
            ys, xs = np.where(im_mask > 0.0)
            if len(xs) == 0:
                pass
            else:
                # 1. Find bounds of white pixels im_mask.
                tl_bound = (np.min(xs), np.min(ys))  # Inclusive
                br_bound = (np.max(xs), np.max(ys))  # Inclusive
                # 2. Mark everything outside the bound as invalid: -1.
                im_mask[:, : tl_bound[0]] = -1.0  # Left
                im_mask[:, br_bound[0] + 1 :] = -1.0  # Right
                im_mask[: tl_bound[1], :] = -1.0  # Top
                im_mask[br_bound[1] + 1 :, :] = -1.0  # Bottom
                # 3. Start from the 4 corners, fill connected components.
                # This will only fill 0 pixels to 1.
                corners = [
                    (tl_bound[0], tl_bound[1]),  # Top-left
                    (br_bound[0], tl_bound[1]),  # Top-right
                    (tl_bound[0], br_bound[1]),  # Bottom-left
                    (br_bound[0], br_bound[1]),  # Bottom-right
                ]
                for corner in corners:
                    im_mask = fill_connected_component(im_mask, corner[0], corner[1])
                # 4. Undo mask invalid pixels.
                im_mask[im_mask == -1.0] = 0.0

        # Draw im_mask on im_dst with the specified color.
        im_dst = np.copy(im)
        color_rgb = matplotlib.colors.to_rgb(edgecolor)
        im_dst[im_mask == 1.0] = color_rgb

        return im_dst

    def _overlay_rectangle_on_image_dashed(
        self,
        im: np.ndarray,
        tl_xy: Tuple[int, int],
        br_xy: Tuple[int, int],
        linewidth_px: int,
        edgecolor: str,
        dash_length: int = None,
    ) -> np.ndarray:
        """
        Draw dashed rectangular bounding box on image using OpenCV.

        Args:
            im: Image to draw bounding box on. Must be float32.
            tl_xy: Top-left corner of bounding box, in (x, y).
            br_xy: Bottom-right corner of bounding box, in (x, y).
            linewidth_px: Width of bounding box line, in pixels.
            edgecolor: Color of bounding box line.
            dash_length: Length of each dash in pixels.
        """
        if dash_length is None:
            dash_length = self.config.dash_length

        if im.dtype != np.float32:
            raise ValueError(f"Invalid image dtype {im.dtype}.")

        im_dst = np.copy(im)
        color_rgb = matplotlib.colors.to_rgb(edgecolor)
        color_bgr = tuple(reversed(color_rgb))

        # Convert to uint8 for OpenCV drawing
        im_uint8 = (im_dst * 255).astype(np.uint8)

        x0, y0 = tl_xy
        x1, y1 = br_xy

        # Draw four sides with dashes
        sides = [
            ((x0, y0), (x1, y0)),  # Top
            ((x1, y0), (x1, y1)),  # Right
            ((x1, y1), (x0, y1)),  # Bottom
            ((x0, y1), (x0, y0)),  # Left
        ]

        for (sx, sy), (ex, ey) in sides:
            length = int(np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2))
            if length == 0:
                continue

            # Draw dashes along the line
            for i in range(0, length, dash_length * 2):
                # Calculate start and end points of this dash
                t_start = i / float(length)
                t_end = min((i + dash_length) / float(length), 1.0)

                dash_sx = int(sx + t_start * (ex - sx))
                dash_sy = int(sy + t_start * (ey - sy))
                dash_ex = int(sx + t_end * (ex - sx))
                dash_ey = int(sy + t_end * (ey - sy))

                cv2.line(
                    im_uint8,
                    (dash_sx, dash_sy),
                    (dash_ex, dash_ey),
                    tuple(int(c * 255) for c in color_rgb),
                    linewidth_px,
                )

        # Convert back to float32
        im_dst = im_uint8.astype(np.float32) / 255.0
        return im_dst

    def _add_enlarged_view(
        self,
        im: np.ndarray,
        im_original: np.ndarray,
        tl_xy: Tuple[int, int],
        br_xy: Tuple[int, int],
        linewidth_px: int,
    ) -> np.ndarray:
        """
        Add an enlarged view of the selected area in the opposite quadrant.

        Args:
            im: Image with dashed bbox already drawn. Must be float32.
            im_original: Original image without any bbox drawn. Must be float32.
            tl_xy: Top-left corner of selected area, in (x, y).
            br_xy: Bottom-right corner of selected area, in (x, y).
            linewidth_px: Width of bounding box line, in pixels.

        Returns:
            Image with enlarged view added.
        """
        h, w = im.shape[:2]
        x0, y0 = tl_xy
        x1, y1 = br_xy

        # Extract the selected area from the ORIGINAL image (without dashed lines)
        crop_x0 = max(0, x0)
        crop_y0 = max(0, y0)
        crop_x1 = min(w, x1)
        crop_y1 = min(h, y1)

        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            return im  # Invalid crop, return original

        cropped = im_original[crop_y0:crop_y1, crop_x0:crop_x1].copy()
        crop_h, crop_w = cropped.shape[:2]

        # Resize to target size based on self.enlarged_view_scale.
        # The enlarged view's shorter side should match enlarged_view_scale.
        crop_shorter = min(crop_h, crop_w)
        if crop_shorter == 0:
            return im

        scale_factor = self.enlarged_view_scale / float(crop_shorter)
        enlarged_w = int(crop_w * scale_factor)
        enlarged_h = int(crop_h * scale_factor)

        # Resize the cropped area
        enlarged = cv2.resize(
            cropped, (enlarged_w, enlarged_h), interpolation=cv2.INTER_LINEAR
        )

        # Determine which quadrant the selected area center is in
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2

        in_left = center_x < w / 2
        in_top = center_y < h / 2

        # Place enlarged view in opposite quadrant (at corner)
        margin = self.config.enlarged_view_margin

        if in_left and in_top:
            # Selected area is top-left → place enlarged view in bottom-right
            paste_x = w - enlarged_w - margin
            paste_y = h - enlarged_h - margin
        elif not in_left and in_top:
            # Selected area is top-right → place enlarged view in bottom-left
            paste_x = margin
            paste_y = h - enlarged_h - margin
        elif in_left and not in_top:
            # Selected area is bottom-left → place enlarged view in top-right
            paste_x = w - enlarged_w - margin
            paste_y = margin
        else:
            # Selected area is bottom-right → place enlarged view in top-left
            paste_x = margin
            paste_y = margin

        # Ensure paste coordinates are within bounds
        paste_x = max(0, min(paste_x, w - enlarged_w))
        paste_y = max(0, min(paste_y, h - enlarged_h))

        # Paste the enlarged view onto the image
        im_result = np.copy(im)
        im_result[paste_y : paste_y + enlarged_h, paste_x : paste_x + enlarged_w] = (
            enlarged
        )

        # Draw solid red border around the enlarged view
        enlarged_tl = (paste_x, paste_y)
        enlarged_br = (paste_x + enlarged_w, paste_y + enlarged_h)
        im_result = BBoxer._overlay_rectangle_on_image(
            im=im_result,
            tl_xy=enlarged_tl,
            br_xy=enlarged_br,
            linewidth_px=linewidth_px,
            edgecolor="red",
            squarecorners=True,
        )

        return im_result

    def _generate_current_images(self):
        """
        Generate the current rendered images based on the current bbox and settings.
        This is the single source of truth for how images are rendered.

        The generated images are stored in self.current_rendered_images and are:
        1. Displayed in the preview
        2. Saved to disk (exact same images)
        """
        # Get linewidth in pixels for the actual image.
        im_height = self.original_images[0].shape[0]
        axis = self.axes[0]
        bbox = axis.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        axis_height = bbox.height * self.fig.dpi
        linewidth_px = self.linewidth * self.fig.dpi / 72.0
        linewidth_px = linewidth_px / axis_height * im_height
        linewidth_px = int(round(linewidth_px))

        # Clear previous rendered images.
        self.current_rendered_images = []

        # Render each image with the bounding box (if any).
        for im_original in self.original_images:
            if self.current_bbox is None:
                # No bbox - show original image.
                im_rendered = im_original
            else:
                x0, y0, x1, y1 = self.current_bbox
                tl_xy = (int(round(x0)), int(round(y0)))
                br_xy = (int(round(x1)), int(round(y1)))

                if self.enlarged_view_mode:
                    # Draw dashed lines for the selected area (area being zoomed).
                    im_rendered = self._overlay_rectangle_on_image_dashed(
                        im=im_original,
                        tl_xy=tl_xy,
                        br_xy=br_xy,
                        linewidth_px=linewidth_px,
                        edgecolor=self.edgecolor,
                    )

                    # Create enlarged view and place in opposite quadrant.
                    # Pass original image so the enlarged view doesn't include dashed lines.
                    im_rendered = self._add_enlarged_view(
                        im_rendered, im_original, tl_xy, br_xy, linewidth_px
                    )
                else:
                    # Draw solid lines (final output mode).
                    im_rendered = BBoxer._overlay_rectangle_on_image(
                        im=im_original,
                        tl_xy=tl_xy,
                        br_xy=br_xy,
                        linewidth_px=linewidth_px,
                        edgecolor=self.edgecolor,
                    )

            self.current_rendered_images.append(im_rendered)

    def _redraw(self):
        """
        Redraw all images by generating current images and displaying them.
        The displayed images are exactly what will be saved.
        """
        # Generate the current rendered images (single source of truth).
        self._generate_current_images()

        # Update the displayed images.
        for img_obj, im_rendered in zip(self.axis_images, self.current_rendered_images):
            img_obj.set_data(im_rendered)

        # Redraw the canvas.
        self.fig.canvas.draw()

    def _save(self) -> None:
        """
        Save the current rendered images to disk.

        The saved images are exactly what's shown in the preview (WYSIWYG).
        No re-rendering is done here - we just save the pre-rendered images.
        """
        if self.current_bbox is None:
            print("No bounding box to save. Exiting.")
            return

        if not self.current_rendered_images:
            print("No rendered images to save. Exiting.")
            return

        # Save each rendered image.
        dst_paths = [p.parent / f"bbox_{p.stem}{p.suffix}" for p in self.src_paths]

        for im_rendered, dst_path in zip(self.current_rendered_images, dst_paths):
            ct.io.imwrite(dst_path, im_rendered)
            print(f"Saved {dst_path}")

    def _print_key(self, key):
        """
        Print keypress event in formatted way.
        """
        # Change the first letter to upper case.
        key = key[0].upper() + key[1:]
        print(f'[KeyPress] "{key}".')

    def _print_button(self, button):
        """
        Print button press event in formatted way.
        """
        # Change the first letter to upper case.
        button = button[0].upper() + button[1:]
        print(f'[ButtonPress] "{button}".')

    def _print_msg(self, *args, prefix="[KeyPress] ", **kwargs):
        """
        Print message with proper indentation to align with keypress/buttonpress output.
        """
        # Simulate sprintf
        string_io = io.StringIO()
        print(*args, file=string_io, **kwargs)
        msg = string_io.getvalue()
        string_io.close()
        prefix_spaces = " " * len(prefix)
        print(f"{prefix_spaces}{msg}", end="")

    def _on_keypress(self, event):
        """
        Callback function for keypress.
        """
        sys.stdout.flush()

        if event.key == "enter":
            self._print_key(event.key)
            if self.current_bbox is None:
                self._print_msg("No bounding box to save.")
            else:
                # Save and quit.
                x0, y0, x1, y1 = self.current_bbox
                bbox_str = f"Bbox({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})"
                self._print_msg(f"Saving bounding box: {bbox_str}.")
                self._close()
        elif event.key == "backspace":
            self._print_key(event.key)
            if self.current_bbox is not None:
                x0, y0, x1, y1 = self.current_bbox
                bbox_str = f"Bbox({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})"
                self.current_bbox = None
                # Hide all rectangle selectors.
                for axis in self.axes:
                    self.axis_to_selector[axis].set_visible(False)
                self._print_msg(f"Bounding box removed: {bbox_str}")
                self._redraw()
            else:
                self._print_msg("No bounding box to remove.")
        elif event.key == "+" or event.key == "=":
            self._print_key(event.key)
            self.linewidth += 1
            self._print_msg(f"Line width increased to: {self.linewidth}")
            self._redraw()
        elif event.key == "-" or event.key == "_":
            self._print_key(event.key)
            if self.linewidth > 1:
                self.linewidth -= 1
                self._print_msg(f"Line width decreased to: {self.linewidth}")
            else:
                self._print_msg(f"Line width already at minimum: {self.linewidth}")
            self._redraw()
        elif event.key == "escape":
            self._print_key(event.key)
            self._close()
        elif event.key == "s":
            self._print_key(event.key)
            self._toggle_square_mode()
        elif event.key == "e":
            self._print_key(event.key)
            self._toggle_enlarged_view()

    def _close(self):
        """
        Close the matplotlib window. This will trigger the _on_close callback.
        """
        plt.close(self.fig)

    def _on_close(self, event):
        """
        Callback function on matplotlib window close.
        """
        print("Closing...")
        self._save()

    def _on_mouse_motion(self, event):
        """
        Callback function for mouse motion events.
        Used to enforce square aspect ratio during interactive resizing.
        """
        if not self.square_mode:
            return

        # Check if any selector is active (being dragged)
        for axis, selector in self.axis_to_selector.items():
            if selector.active and event.inaxes == axis:
                # Get current extents
                extents = selector.extents  # (x0, x1, y0, y1)
                x0, x1, y0, y1 = extents

                width = x1 - x0
                height = y1 - y0
                size = min(abs(width), abs(height))

                if size < 1e-6:
                    return

                # Preserve top-left corner, adjust bottom-right to maintain square
                new_x1 = x0 + size if width >= 0 else x0 - size
                new_y1 = y0 + size if height >= 0 else y0 - size

                # Update selector extents
                try:
                    selector.extents = (x0, new_x1, y0, new_y1)
                except:
                    pass

                break

    def _increase_linewidth(self, event):
        """
        Callback function for increase line width button.
        """
        sys.stdout.flush()
        self._print_button("+")
        self.linewidth += 1
        self._print_msg(
            f"Line width increased to: {self.linewidth}", prefix="[ButtonPress] "
        )
        self._redraw()

    def _decrease_linewidth(self, event):
        """
        Callback function for decrease line width button.
        """
        sys.stdout.flush()
        self._print_button("-")
        if self.linewidth > 1:
            self.linewidth -= 1
            self._print_msg(
                f"Line width decreased to: {self.linewidth}", prefix="[ButtonPress] "
            )
        else:
            self._print_msg(
                f"Line width already at minimum: {self.linewidth}",
                prefix="[ButtonPress] ",
            )
        self._redraw()

    def _increase_enlarged_size(self, event):
        """
        Callback function for increase enlarged view size button.
        """
        sys.stdout.flush()
        self._print_button("Enlarged +")
        step = int(self.enlarged_view_scale_max * self.config.enlarged_view_step_ratio)
        if self.enlarged_view_scale < self.enlarged_view_scale_max:
            self.enlarged_view_scale = min(
                self.enlarged_view_scale + step, self.enlarged_view_scale_max
            )
            self._print_msg(
                f"Enlarged view size increased to: {self.enlarged_view_scale}px",
                prefix="[ButtonPress] ",
            )
            self._redraw()
        else:
            self._print_msg(
                f"Enlarged view size already at maximum: {self.enlarged_view_scale}px",
                prefix="[ButtonPress] ",
            )

    def _decrease_enlarged_size(self, event):
        """
        Callback function for decrease enlarged view size button.
        """
        sys.stdout.flush()
        self._print_button("Enlarged -")
        step = int(self.enlarged_view_scale_max * self.config.enlarged_view_step_ratio)
        min_size = int(
            self.enlarged_view_scale_max * self.config.enlarged_view_min_ratio
        )
        if self.enlarged_view_scale > min_size:
            self.enlarged_view_scale = max(self.enlarged_view_scale - step, min_size)
            self._print_msg(
                f"Enlarged view size decreased to: {self.enlarged_view_scale}px",
                prefix="[ButtonPress] ",
            )
            self._redraw()
        else:
            self._print_msg(
                f"Enlarged view size already at minimum: {self.enlarged_view_scale}px",
                prefix="[ButtonPress] ",
            )

    def _create_buttons(self):
        """
        Create all interactive buttons in a clean layout.
        """
        # Adjust layout to make room for buttons at the bottom.
        self.fig.subplots_adjust(bottom=self.config.bottom_margin)

        # Button dimensions from config.
        btn_h = self.config.button_height
        btn_small = self.config.button_small_width
        btn_medium = self.config.button_medium_width
        btn_font_size = self.config.button_font_size
        spacing = self.config.button_spacing
        y_row1 = self.config.button_row1_y
        y_row2 = self.config.button_row2_y

        # Calculate total width and center all buttons.
        # Row 1: Line width label + +/- + spacing + Enlarged size label + +/-
        row1_width = (
            0.08 + btn_small * 2 + spacing * 3 + 0.10 + btn_small * 2 + spacing * 2
        )
        # Row 2: Square button + spacing + Enlarged button
        row2_width = btn_medium * 2 + spacing

        # Start positions (centered).
        row1_start = 0.5 - row1_width / 2
        row2_start = 0.5 - row2_width / 2

        # --- ROW 1: Line Width and Enlarged Size Controls ---
        x = row1_start

        # Line width label.
        ax_lw_label = self.fig.add_axes([x, y_row1, 0.08, btn_h])
        ax_lw_label.text(
            0.5,
            0.5,
            "Line:",
            ha="center",
            va="center",
            transform=ax_lw_label.transAxes,
            fontsize=self.config.button_label_font_size,
        )
        ax_lw_label.set_axis_off()
        x += 0.08 + spacing

        # Line width decrease button.
        ax_lw_dec = self.fig.add_axes([x, y_row1, btn_small, btn_h])
        self.button_linewidth_decrease = Button(ax_lw_dec, "-")
        self.button_linewidth_decrease.label.set_fontsize(btn_font_size)
        self.button_linewidth_decrease.on_clicked(self._decrease_linewidth)
        x += btn_small + spacing

        # Line width increase button.
        ax_lw_inc = self.fig.add_axes([x, y_row1, btn_small, btn_h])
        self.button_linewidth_increase = Button(ax_lw_inc, "+")
        self.button_linewidth_increase.label.set_fontsize(btn_font_size)
        self.button_linewidth_increase.on_clicked(self._increase_linewidth)
        x += btn_small + spacing * 3

        # Enlarged size label.
        ax_es_label = self.fig.add_axes([x, y_row1, 0.10, btn_h])
        ax_es_label.text(
            0.5,
            0.5,
            "Zoom:",
            ha="center",
            va="center",
            transform=ax_es_label.transAxes,
            fontsize=self.config.button_label_font_size,
        )
        ax_es_label.set_axis_off()
        x += 0.10 + spacing

        # Enlarged size decrease button.
        ax_es_dec = self.fig.add_axes([x, y_row1, btn_small, btn_h])
        self.button_enlarged_size_decrease = Button(ax_es_dec, "-")
        self.button_enlarged_size_decrease.label.set_fontsize(btn_font_size)
        self.button_enlarged_size_decrease.on_clicked(self._decrease_enlarged_size)
        x += btn_small + spacing

        # Enlarged size increase button.
        ax_es_inc = self.fig.add_axes([x, y_row1, btn_small, btn_h])
        self.button_enlarged_size_increase = Button(ax_es_inc, "+")
        self.button_enlarged_size_increase.label.set_fontsize(btn_font_size)
        self.button_enlarged_size_increase.on_clicked(self._increase_enlarged_size)

        # --- ROW 2: Toggle Buttons ---
        x = row2_start

        # Square mode button.
        ax_square = self.fig.add_axes([x, y_row2, btn_medium, btn_h])
        self.button_square_mode = Button(ax_square, "Square: ON")
        self.button_square_mode.label.set_fontsize(btn_font_size)
        self.button_square_mode.on_clicked(self._on_square_mode_button_click)
        x += btn_medium + spacing

        # Enlarged view button.
        ax_enlarged = self.fig.add_axes([x, y_row2, btn_medium, btn_h])
        self.button_enlarged_view = Button(ax_enlarged, "Enlarged: ON")
        self.button_enlarged_view.label.set_fontsize(btn_font_size)
        self.button_enlarged_view.on_clicked(self._on_enlarged_view_button_click)

    def _toggle_square_mode(self):
        """
        Toggle square mode on/off.
        """
        self.square_mode = not self.square_mode

        # If toggling to square mode and there's a current bbox, convert it to square.
        if self.square_mode and self.current_bbox is not None:
            x0, y0, x1, y1 = self.current_bbox

            width = x1 - x0
            height = y1 - y0
            size = min(abs(width), abs(height))

            # Center the square on the original bbox
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2

            new_x0 = center_x - size / 2
            new_y0 = center_y - size / 2
            new_x1 = center_x + size / 2
            new_y1 = center_y + size / 2

            self.current_bbox = (new_x0, new_y0, new_x1, new_y1)

            # Synchronize the selector's extents with the new square bbox
            for axis, selector in self.axis_to_selector.items():
                if selector.get_visible():
                    try:
                        selector.extents = (new_x0, new_x1, new_y0, new_y1)
                        selector.update()
                    except:
                        pass
                    break

            self._redraw()

        # Update button label.
        if self.button_square_mode is not None:
            self.button_square_mode.label.set_text(
                "Square: ON" if self.square_mode else "Square: OFF"
            )
            self.fig.canvas.draw_idle()

        mode_str = "enabled" if self.square_mode else "disabled"
        self._print_msg(f"Square mode {mode_str}")

    def _toggle_enlarged_view(self):
        """
        Toggle enlarged view mode on/off.
        """
        self.enlarged_view_mode = not self.enlarged_view_mode

        # Update button label.
        if self.button_enlarged_view is not None:
            self.button_enlarged_view.label.set_text(
                "Enlarged: ON" if self.enlarged_view_mode else "Enlarged: OFF"
            )
            self.fig.canvas.draw_idle()

        mode_str = "enabled" if self.enlarged_view_mode else "disabled"
        self._print_msg(f"Enlarged view mode {mode_str}")

        # Redraw to show/hide enlarged view
        self._redraw()

    def _on_square_mode_button_click(self, event):
        """
        Callback function for square mode toggle button.
        """
        sys.stdout.flush()
        self._print_button("Square mode")
        self._toggle_square_mode()

    def _on_enlarged_view_button_click(self, event):
        """
        Callback function for enlarged view toggle button.
        """
        sys.stdout.flush()
        self._print_button("Enlarged view")
        self._toggle_enlarged_view()

    def _register_rectangle_selector(self, axis):
        """
        Register "on selector" event handler for a given axis.
        """

        def on_selector(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            # If square mode is enabled, enforce square aspect ratio.
            if self.square_mode:
                width = np.abs(x2 - x1)
                height = np.abs(y2 - y1)
                size = min(width, height)

                # Determine center point based on which direction we're dragging
                if x2 > x1:
                    center_x = x1 + width / 2
                else:
                    center_x = x1 - width / 2

                if y2 > y1:
                    center_y = y1 + height / 2
                else:
                    center_y = y1 - height / 2

                # Create square bbox centered on the drag
                x0 = center_x - size / 2
                y0 = center_y - size / 2
                x1_new = center_x + size / 2
                y1_new = center_y + size / 2

                self.current_bbox = (x0, y0, x1_new, y1_new)
            else:
                # Create bbox from drag coordinates
                x0 = min(x1, x2)
                y0 = min(y1, y2)
                x1_new = max(x1, x2)
                y1_new = max(y1, y2)

                self.current_bbox = (x0, y0, x1_new, y1_new)

            # Hide other selectors.
            current_axis = eclick.inaxes
            for axis in self.axes:
                if axis != current_axis:
                    self.axis_to_selector[axis].set_visible(False)

            # Redraw with the new bbox using OpenCV rendering.
            self._redraw()

        # If not saved, the selector will go out-of-scope.
        self.axis_to_selector[axis] = RectangleSelector(
            axis,
            on_selector,
            useblit=False,
            button=[1],
            minspanx=self.config.min_span_pixels,
            minspany=self.config.min_span_pixels,
            spancoords="pixels",
            interactive=True,
        )

    @staticmethod
    def _print_help_message():
        """
        Print help messages of keyboard callbacks.
        """
        print("Drag     : Draw bounding box (release mouse to see result).")
        print("Enter    : Save bounding box and quit.")
        print("Backspace: Remove current bounding box.")
        print("+ or =   : Increase line width (or use + button).")
        print("- or _   : Decrease line width (or use - button).")
        print("s        : Toggle square mode (or use Square button, default: ON).")
        print("e        : Toggle enlarged view (or use Enlarged button, default: ON).")
        print("Escape   : Quit without saving.")


def instantiate_parser(parser):
    parser.add_argument(
        "inputs",
        type=Path,
        help="Input image paths.",
        nargs="+",
    )
    return parser


def entry_point(parser, args):
    """
    This is used by sub_parser.set_defaults(func=entry_point).
    The parser argument is not used.
    """

    if isinstance(args.inputs, list):
        src_paths = args.inputs
    elif isinstance(args.inputs, str):
        src_paths = [args.inputs]
    else:
        raise ValueError(f"Invalid input type: {type(args.inputs)}")

    bboxer = BBoxer()
    bboxer.add_paths(src_paths)
    bboxer.run()


def main():
    # Usage 1:
    # ct draw-bboxes camtools/assets/box.png camtools/assets/box_blender.png

    # Usage 2:
    # python camtools/tools/draw_bbox.py camtools/assets/box_blender.png
    parser = argparse.ArgumentParser()
    parser = instantiate_parser(parser)
    args = parser.parse_args()
    entry_point(parser, args)

    # Or, you can import the class and use it as a library.
    # camtools_dir = Path(__file__).parent.parent.absolute()
    # bboxer = BBoxer()
    # bboxer.add_paths([
    #     camtools_dir / "assets" / "box.png",
    #     camtools_dir / "assets" / "box_blender.png",
    # ])
    # bboxer.run()


if __name__ == "__main__":
    main()
