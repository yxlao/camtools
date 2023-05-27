from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.widgets import RectangleSelector
import sys
import camtools as ct
from typing import List
import tempfile
from pathlib import Path
import io


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
        self.current_rec = None
        self.confirmed_recs = []
        self.visible_recs = []

        # Other matplotlib objects.
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
            1. Load images.
            2. Display images simultaneously, side-by-side.
            3. Interactively draw bounding boxes on images.
            4. Save images with bounding boxes to disk.

        Notes:
            1. Currently the output path is hardcoded.
            2. The input images must have the same dimensions and 3 channels.
        """
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
                raise ValueError("Images must have the same shape "
                                 f"{im_src.shape} != {im_srcs[0].shape}")

        # Register fig and axes.
        self.fig, self.axes = plt.subplots(1, len(im_srcs))
        for i, (axis, im_src) in enumerate(zip(self.axes, im_srcs)):
            axis.imshow(im_src)
            axis.set_title(self.src_paths[i].name)
            axis.set_axis_off()
        plt.tight_layout()

        # Register rectangle selector callbacks.
        for axis in self.axes:
            self._register_rectangle_selector(axis)

        # Register other handlers.
        self.fig.canvas.mpl_connect('key_press_event', self._on_keypress)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        plt.show()

    @staticmethod
    def _bbox_str(bbox: matplotlib.transforms.Bbox) -> str:
        return f"Bbox({bbox.x0:.2f}, {bbox.y0:.2f}, {bbox.x1:.2f}, {bbox.y1:.2f})"

    @staticmethod
    def _copy_rec(rec: matplotlib.patches.Rectangle,
                  linestyle: str = None,
                  linewidth: int = None,
                  edgecolor=None) -> matplotlib.patches.Rectangle:
        new_rec = matplotlib.patches.Rectangle(
            xy=(rec.xy[0], rec.xy[1]),
            width=rec.get_width(),
            height=rec.get_height(),
            linestyle=linestyle
            if linestyle is not None else rec.get_linestyle(),
            linewidth=linewidth
            if linewidth is not None else rec.get_linewidth(),
            edgecolor=edgecolor
            if edgecolor is not None else rec.get_edgecolor(),
            facecolor=rec.get_facecolor(),
        )
        return new_rec

    @staticmethod
    def _overlay_bbox_on_image(
        im: np.ndarray,
        bboxes: List[matplotlib.patches.Rectangle],
        linewidth: int,
        edgecolor: str,
    ) -> np.ndarray:
        """
        Draw red rectangular bounding box on image.

        Args:
            im: Image to draw bounding box on.
            bboxes: List of Matplotlib bounding boxes to draw on image.
        """
        fig, axis = plt.subplots()
        axis.set_axis_off()
        axis.imshow(im)
        for bbox in bboxes:
            axis.add_patch(
                BBoxer._copy_rec(bbox,
                                 linestyle="-",
                                 linewidth=linewidth,
                                 edgecolor=edgecolor))
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            plt.savefig(f.name, bbox_inches='tight')
            im_dst = ct.io.imread(f.name, alpha_mode="ignore")
        plt.close()

        return ct.image.crop_white_boarders(im_dst)

    def _redraw(self):
        # Clear all visible rectangles.
        for rec in self.visible_recs:
            rec.remove()
        self.visible_recs.clear()

        # Draw confirmed rectangles.
        for rec in self.confirmed_recs:
            for axis in self.axes:
                rec_ = axis.add_patch(
                    BBoxer._copy_rec(rec,
                                     linestyle="-",
                                     linewidth=self.linewidth,
                                     edgecolor=self.edgecolor))
                self.visible_recs.append(rec_)

        # Draw current rectangle.
        if self.current_rec is not None:
            for axis in self.axes:
                rec_ = axis.add_patch(
                    BBoxer._copy_rec(self.current_rec,
                                     linestyle="--",
                                     linewidth=self.linewidth,
                                     edgecolor=self.edgecolor))
                self.visible_recs.append(rec_)

        # Ask matplotlib to redraw the current figure.
        # No need to call self.fig.canvas.flush_events().
        self.fig.canvas.draw()

    def _save(self) -> None:
        """
        Save images with bounding boxes to disk. This function is called by the
        matplotlib event handler when the figure is closed.

        If self.confirmed_recs is empty, then no bounding boxes will be drawn,
        but the images will still be saved.
        """
        dst_paths = [
            p.parent / f"bbox_{p.stem}{p.suffix}" for p in self.src_paths
        ]
        for src_path, dst_path in zip(self.src_paths, dst_paths):
            im_src = ct.io.imread(src_path)
            im_dst = BBoxer._overlay_bbox_on_image(im_src,
                                                   self.confirmed_recs,
                                                   linewidth=self.linewidth,
                                                   edgecolor=self.edgecolor)
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
            print(f"[Keypress] \"{key}\".")

        def print_msg(*args, **kwargs):
            # Simulate sprintf
            string_io = io.StringIO()
            print(*args, file=string_io, **kwargs)
            msg = string_io.getvalue()
            string_io.close()
            prefix = " " * len("[Keypress] ")
            print(f"{prefix}{msg}", end="")

        # Check if enter is pressed.
        if event.key == "enter":
            print_key(event.key)
            if self.current_rec is None:
                print_msg("No new BBox selected.")
            else:
                current_bbox = self.current_rec.get_bbox()
                bbox_exists = False
                for bbox in self.confirmed_recs:
                    if current_bbox == bbox.get_bbox():
                        bbox_exists = True
                        break
                if bbox_exists:
                    print_msg("BBox already exists. Not saving.")
                else:
                    # Save to confirmed.
                    self.confirmed_recs.append(
                        BBoxer._copy_rec(self.current_rec))
                    bbox_str = BBoxer._bbox_str(self.current_rec.get_bbox())
                    print_msg(f"BBox saved: {bbox_str}.")
                    # Clear current.
                    self.current_rec = None
                    # Hide all rectangle selectors.
                    for axis in self.axes:
                        self.axis_to_selector[axis].set_visible(False)
            self._redraw()

        elif event.key == "backspace":
            print_key(event.key)
            if self.current_rec is not None:
                bbox_str = BBoxer._bbox_str(self.current_rec.get_bbox())
                self.current_rec = None
                # Hide all rectangle selectors.
                for axis in self.axes:
                    self.axis_to_selector[axis].set_visible(False)
                print_msg(f"Current BBox removed: {bbox_str},")
            else:
                if len(self.confirmed_recs) > 0:
                    last_rec = self.confirmed_recs.pop()
                    bbox_str = BBoxer._bbox_str(last_rec.get_bbox())
                    print_msg(f"Last BBox removed: {bbox_str}")
                else:
                    print_msg("No BBox to remove.")
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
        print('Closing...')
        self._save()

    def _register_rectangle_selector(self, axis):
        """
        Register "on selector" event handler for a given axis.
        """

        def on_selector(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            rect = plt.Rectangle(
                (min(x1, x2), min(y1, y2)),
                np.abs(x1 - x2),
                np.abs(y1 - y2),
                linewidth=self.linewidth,
                edgecolor=self.edgecolor,
                facecolor='none',
            )

            # Hide other selectors.
            current_axis = eclick.inaxes
            for axis in self.axes:
                if axis != current_axis:
                    self.axis_to_selector[axis].set_visible(False)

            # Set current rectangle.
            self.current_rec = rect

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
            spancoords='pixels',
            interactive=True,
        )


def main():
    camtools_dir = Path(__file__).parent.parent.absolute()

    bboxer = BBoxer()
    bboxer.add_paths([
        camtools_dir / "assets" / "box.jpg",
        camtools_dir / "assets" / "box_blender.jpg",
    ])
    bboxer.run()


if __name__ == "__main__":
    main()
