import io
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

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

        # Bounding boxes.
        self.current_rectangle = None  # Only one copy of this.
        self.confirmed_rectangles = []  # Only one copy of this.
        self.axes_owned_rectangles = []  # MRectangles are duplicated in axes.

        # Matplotlib objects.
        self.fig = None
        self.axes = []
        self.axis_to_selector = dict()

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
            4. Save images with bounding boxes to disk (with hard-coded paths).
        """
        BBoxer._print_help_message()

        if len(self.src_paths) == 0:
            raise ValueError("No input images.")

        # Load.
        im_srcs = []
        for src_path in self.src_paths:
            im_src = ct.io.imread(src_path)
            if im_src.ndim != 3 or im_src.shape[2] != 3:
                raise ValueError(f"Invalid image shape {im_src.shape}.")
            im_srcs.append(im_src)

        # Check all images are of the same shape.
        for im_src in im_srcs:
            if im_src.shape != im_srcs[0].shape:
                raise ValueError(
                    "Images must have the same shape "
                    f"{im_src.shape} != {im_srcs[0].shape}"
                )

        # Register fig and axes.
        self.fig, self.axes = plt.subplots(1, len(im_srcs))
        if len(im_srcs) == 1:
            self.axes = [self.axes]
        for i, (axis, im_src) in enumerate(zip(self.axes, im_srcs)):
            axis.imshow(im_src)
            axis.set_title(self.src_paths[i].name)
            axis.set_axis_off()
        plt.tight_layout()

        # Register rectangle selector callbacks.
        for axis in self.axes:
            self._register_rectangle_selector(axis)

        # Register other handlers.
        self.fig.canvas.mpl_connect("key_press_event", self._on_keypress)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        plt.show()

    @staticmethod
    def _bbox_str(bbox: matplotlib.transforms.Bbox) -> str:
        """
        A better matplotlib.transforms.Bbox.__str__()`
        """
        return f"Bbox({bbox.x0:.2f}, {bbox.y0:.2f}, {bbox.x1:.2f}, {bbox.y1:.2f})"

    @staticmethod
    def _copy_rectangle(
        rectangle: matplotlib.patches.Rectangle,
        linestyle: str = None,
        linewidth: int = None,
        edgecolor=None,
    ) -> matplotlib.patches.Rectangle:
        """
        Copy rectangle with new properties.
        """
        new_rectangle = matplotlib.patches.Rectangle(
            xy=(rectangle.xy[0], rectangle.xy[1]),
            width=rectangle.get_width(),
            height=rectangle.get_height(),
            linestyle=linestyle if linestyle is not None else rectangle.get_linestyle(),
            linewidth=linewidth if linewidth is not None else rectangle.get_linewidth(),
            edgecolor=edgecolor if edgecolor is not None else rectangle.get_edgecolor(),
            facecolor=rectangle.get_facecolor(),
        )
        return new_rectangle

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
        Triggers redraw of all axis and rectangles.
        """
        # Clear all visible rectangles.
        for rectangle in self.axes_owned_rectangles:
            rectangle.remove()
        self.axes_owned_rectangles.clear()

        # Draw confirmed rectangles.
        for rectangle in self.confirmed_rectangles:
            for axis in self.axes:
                rectangle_ = axis.add_patch(
                    BBoxer._copy_rectangle(
                        rectangle,
                        linestyle="-",
                        linewidth=self.linewidth,
                        edgecolor=self.edgecolor,
                    )
                )
                self.axes_owned_rectangles.append(rectangle_)

        # Draw current rectangle.
        if self.current_rectangle is not None:
            for axis in self.axes:
                rectangle_ = axis.add_patch(
                    BBoxer._copy_rectangle(
                        self.current_rectangle,
                        linestyle="--",
                        linewidth=self.linewidth,
                        edgecolor=self.edgecolor,
                    )
                )
                self.axes_owned_rectangles.append(rectangle_)

        # Ask matplotlib to redraw the current figure.
        # No need to call self.fig.canvas.flush_events().
        self.fig.canvas.draw()

    def _save(self) -> None:
        """
        Save images with bounding boxes to disk. This function is called by the
        matplotlib event handler when the figure is closed.

        If self.confirmed_rectangles is empty, then no image will be saved.
        """
        if len(self.confirmed_rectangles) == 0:
            print("Make sure to press Enter to confirm bounding boxes.")
            print("No bounding box to save. Exiting.")
            return

        # Get the axis image shape in pixels.
        im_shape = self.axes[0].get_images()[0].get_array().shape
        im_height = im_shape[0]

        axis = self.axes[0]
        bbox = axis.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        axis_height = bbox.height * self.fig.dpi

        # Get the linewidth in pixels.
        linewidth_px = self.linewidth * self.fig.dpi / 72.0
        linewidth_px = linewidth_px / axis_height * im_height
        linewidth_px = int(round(linewidth_px))

        dst_paths = [p.parent / f"bbox_{p.stem}{p.suffix}" for p in self.src_paths]
        for src_path, dst_path in zip(self.src_paths, dst_paths):
            im_dst = ct.io.imread(src_path)
            for rectangle in self.confirmed_rectangles:
                bbox = rectangle.get_bbox()
                tl_xy = (int(round(bbox.x0)), int(round(bbox.y0)))
                br_xy = (int(round(bbox.x1)), int(round(bbox.y1)))
                im_dst = BBoxer._overlay_rectangle_on_image(
                    im=im_dst,
                    tl_xy=tl_xy,
                    br_xy=br_xy,
                    linewidth_px=linewidth_px,
                    edgecolor=self.edgecolor,
                )
            ct.io.imwrite(dst_path, im_dst)
            print(f"Saved {dst_path}")

    def _on_keypress(self, event):
        """
        Callback function for keypress.
        """
        sys.stdout.flush()

        def print_key(key):
            # Change the first letter to upper case.
            key = key[0].upper() + key[1:]
            print(f'[Keypress] "{key}".')

        def print_msg(*args, **kwargs):
            # Simulate sprintf
            string_io = io.StringIO()
            print(*args, file=string_io, **kwargs)
            msg = string_io.getvalue()
            string_io.close()
            prefix = " " * len("[Keypress] ")
            print(f"{prefix}{msg}", end="")

        if event.key == "enter":
            print_key(event.key)
            if self.current_rectangle is None:
                print_msg("No new bounding boxes selected.")
            else:
                current_bbox = self.current_rectangle.get_bbox()
                bbox_exists = False
                for bbox in self.confirmed_rectangles:
                    if current_bbox == bbox.get_bbox():
                        bbox_exists = True
                        break
                if bbox_exists:
                    print_msg("Bounding box already exists. Not saving.")
                else:
                    # Save to confirmed.
                    self.confirmed_rectangles.append(
                        BBoxer._copy_rectangle(self.current_rectangle)
                    )
                    bbox_str = BBoxer._bbox_str(self.current_rectangle.get_bbox())
                    print_msg(f"Bounding box saved: {bbox_str}.")
                    # Clear current.
                    self.current_rectangle = None
                    # Hide all rectangle selectors.
                    for axis in self.axes:
                        self.axis_to_selector[axis].set_visible(False)
            self._redraw()
        elif event.key == "backspace":
            print_key(event.key)
            if self.current_rectangle is not None:
                bbox_str = BBoxer._bbox_str(self.current_rectangle.get_bbox())
                self.current_rectangle = None
                # Hide all rectangle selectors.
                for axis in self.axes:
                    self.axis_to_selector[axis].set_visible(False)
                print_msg(f"Current bounding box removed: {bbox_str},")
            else:
                if len(self.confirmed_rectangles) > 0:
                    last_rectangle = self.confirmed_rectangles.pop()
                    bbox_str = BBoxer._bbox_str(last_rectangle.get_bbox())
                    print_msg(f"Last bounding box removed: {bbox_str}")
                else:
                    print_msg("No bounding box to remove.")
            self._redraw()
        elif event.key == "+" or event.key == "=":
            print_key(event.key)
            self.linewidth += 1
            print_msg(f"Line width increased to: {self.linewidth}")
            self._redraw()
        elif event.key == "-" or event.key == "_":
            print_key(event.key)
            if self.linewidth > 1:
                self.linewidth -= 1
                print_msg(f"Line width decreased to: {self.linewidth}")
            else:
                print_msg(f"Line width already at minimum: {self.linewidth}")
            self._redraw()
        elif event.key == "escape":
            print_key(event.key)
            self._close()

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

    def _register_rectangle_selector(self, axis):
        """
        Register "on selector" event handler for a given axis.
        """

        def on_selector(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            rectanglet = plt.Rectangle(
                (min(x1, x2), min(y1, y2)),
                np.abs(x1 - x2),
                np.abs(y1 - y2),
                linewidth=self.linewidth,
                edgecolor=self.edgecolor,
                facecolor="none",
            )

            # Hide other selectors.
            current_axis = eclick.inaxes
            for axis in self.axes:
                if axis != current_axis:
                    self.axis_to_selector[axis].set_visible(False)

            # Set current rectangle.
            self.current_rectangle = rectanglet

            # Draw current rectangle and confirmed rectangles.
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
        print("Enter    : Save current bounding box.")
        print("Backspace: Remove current bounding box.")
        print("+ or =   : Increase line width.")
        print("- or _   : Decrease line width.")


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
