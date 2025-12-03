import io
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector, Button

import camtools as ct
import argparse


class BBoxer:
    """
    Draw bounding boxes on images.
    """

    def __init__(self, linewidth=1, edgecolor="red"):
        # Draw properties.
        self.linewidth = linewidth
        self.edgecolor = edgecolor

        # Input image paths.
        self.src_paths: List[Path] = []

        # Original images (never modified).
        self.original_images = []

        # Bounding box - only one at a time.
        self.current_bbox = None  # Tuple: (x0, y0, x1, y1) or None

        # Matplotlib objects.
        self.fig = None
        self.axes = []
        self.axis_images = []  # Store image objects for updating
        self.axis_to_selector = dict()
        self.button_increase = None
        self.button_decrease = None
        self.button_square_mode = None

        # Square mode state.
        self.square_mode = False

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

        # Register fig and axes.
        self.fig, self.axes = plt.subplots(1, len(self.original_images))
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

        # Create interactive buttons for line width control.
        self._create_linewidth_buttons()

        # Create square mode toggle button.
        self._create_square_mode_button()

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

    def _redraw(self):
        """
        Redraw all images with the current bounding box using OpenCV rendering.
        This shows the exact final output that will be saved.
        """
        # Get linewidth in pixels for the actual image.
        im_height = self.original_images[0].shape[0]
        axis = self.axes[0]
        bbox = axis.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        axis_height = bbox.height * self.fig.dpi
        linewidth_px = self.linewidth * self.fig.dpi / 72.0
        linewidth_px = linewidth_px / axis_height * im_height
        linewidth_px = int(round(linewidth_px))

        # Render each image with the bounding box (if any).
        for i, (img_obj, im_original) in enumerate(
            zip(self.axis_images, self.original_images)
        ):
            if self.current_bbox is None:
                # No bbox - show original image.
                im_display = im_original
            else:
                # Draw bbox on image using OpenCV.
                x0, y0, x1, y1 = self.current_bbox
                tl_xy = (int(round(x0)), int(round(y0)))
                br_xy = (int(round(x1)), int(round(y1)))
                im_display = BBoxer._overlay_rectangle_on_image(
                    im=im_original,
                    tl_xy=tl_xy,
                    br_xy=br_xy,
                    linewidth_px=linewidth_px,
                    edgecolor=self.edgecolor,
                )

            # Update the displayed image.
            img_obj.set_data(im_display)

        # Redraw the canvas.
        self.fig.canvas.draw()

    def _save(self) -> None:
        """
        Save images with bounding box to disk.

        If no bounding box is defined, no image will be saved.
        """
        if self.current_bbox is None:
            print("No bounding box to save. Exiting.")
            return

        # Get linewidth in pixels.
        im_height = self.original_images[0].shape[0]
        axis = self.axes[0]
        bbox = axis.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        axis_height = bbox.height * self.fig.dpi
        linewidth_px = self.linewidth * self.fig.dpi / 72.0
        linewidth_px = linewidth_px / axis_height * im_height
        linewidth_px = int(round(linewidth_px))

        # Save each image with the bounding box.
        dst_paths = [p.parent / f"bbox_{p.stem}{p.suffix}" for p in self.src_paths]
        x0, y0, x1, y1 = self.current_bbox
        tl_xy = (int(round(x0)), int(round(y0)))
        br_xy = (int(round(x1)), int(round(y1)))

        for im_original, dst_path in zip(self.original_images, dst_paths):
            im_dst = BBoxer._overlay_rectangle_on_image(
                im=im_original,
                tl_xy=tl_xy,
                br_xy=br_xy,
                linewidth_px=linewidth_px,
                edgecolor=self.edgecolor,
            )
            ct.io.imwrite(dst_path, im_dst)
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

    def _create_linewidth_buttons(self):
        """
        Create interactive buttons for increasing and decreasing line width.
        """
        # Position buttons at the bottom of the figure.
        # Adjust layout to make room for buttons.
        self.fig.subplots_adjust(bottom=0.1)

        # Button dimensions and positions (in figure coordinates).
        button_width = 0.08
        button_height = 0.04
        button_y = 0.02
        button_spacing = 0.02
        label_width = 0.12
        label_spacing = 0.02

        # Calculate button positions (centered horizontally).
        num_buttons = 2
        total_width = (
            label_width
            + label_spacing
            + num_buttons * button_width
            + (num_buttons - 1) * button_spacing
        )
        start_x = 0.5 - total_width / 2

        # Create text label.
        ax_label = self.fig.add_axes([start_x, button_y, label_width, button_height])
        ax_label.text(
            0.5,
            0.5,
            "Line width:",
            ha="center",
            va="center",
            transform=ax_label.transAxes,
            fontsize=10,
        )
        ax_label.set_axis_off()

        # Create increase button.
        ax_increase = self.fig.add_axes(
            [
                start_x + label_width + label_spacing,
                button_y,
                button_width,
                button_height,
            ]
        )
        self.button_increase = Button(ax_increase, "+")
        self.button_increase.on_clicked(self._increase_linewidth)

        # Create decrease button.
        ax_decrease = self.fig.add_axes(
            [
                start_x + label_width + label_spacing + button_width + button_spacing,
                button_y,
                button_width,
                button_height,
            ]
        )
        self.button_decrease = Button(ax_decrease, "-")
        self.button_decrease.on_clicked(self._decrease_linewidth)

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

    def _on_square_mode_button_click(self, event):
        """
        Callback function for square mode toggle button.
        """
        sys.stdout.flush()
        self._print_button("Square mode")
        self._toggle_square_mode()

    def _create_square_mode_button(self):
        """
        Create interactive button for toggling square mode.
        """
        # Calculate the end position of line width buttons.
        # This matches the calculation in _create_linewidth_buttons.
        label_width = 0.12
        label_spacing = 0.02
        linewidth_button_width = 0.08
        linewidth_button_spacing = 0.02
        num_linewidth_buttons = 2

        total_linewidth_width = (
            label_width
            + label_spacing
            + num_linewidth_buttons * linewidth_button_width
            + (num_linewidth_buttons - 1) * linewidth_button_spacing
        )
        linewidth_start_x = 0.5 - total_linewidth_width / 2
        linewidth_end_x = linewidth_start_x + total_linewidth_width

        # Square mode button dimensions.
        button_width = 0.16  # Made wider to fit "Square: ON/OFF" text
        button_height = 0.04
        button_y = 0.02

        # Add padding between line width buttons and square button.
        padding = 0.04
        button_x = linewidth_end_x + padding

        # Create square mode button.
        ax_square = self.fig.add_axes([button_x, button_y, button_width, button_height])
        self.button_square_mode = Button(ax_square, "Square: OFF")
        self.button_square_mode.on_clicked(self._on_square_mode_button_click)

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
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

    @staticmethod
    def _print_help_message():
        """
        Print help messages of keyboard callbacks.
        """
        print("Drag     : Draw bounding box (releases mouse to see final result).")
        print("Enter    : Save bounding box and quit.")
        print("Backspace: Remove current bounding box.")
        print("+ or =   : Increase line width (or use + button).")
        print("- or _   : Decrease line width (or use - button).")
        print("s        : Toggle square mode (or use Square button).")
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
