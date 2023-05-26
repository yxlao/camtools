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


class BBoxer:
    """
    Draw bounding boxes on images.
    """

    def __init__(self, line_width=1, edge_color="r"):
        self.line_width = line_width
        self.edge_color = edge_color

        self.src_paths: List[Path] = []
        self.current_patch = None
        self.confirm_patches = []

        self.visible_patches = []

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

    @staticmethod
    def _copy_bbox(
            bbox: matplotlib.patches.Rectangle) -> matplotlib.patches.Rectangle:
        new_bbox = matplotlib.patches.Rectangle(
            xy=(bbox.xy[0], bbox.xy[1]),
            width=bbox.get_width(),
            height=bbox.get_height(),
            linewidth=bbox.get_linewidth(),
            edgecolor=bbox.get_edgecolor(),
            facecolor=bbox.get_facecolor(),
        )
        return new_bbox

    @staticmethod
    def _overlay_bbox_on_image(
            im: np.ndarray,
            bboxes: List[matplotlib.patches.Rectangle]) -> np.ndarray:
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
            axis.add_patch(BBoxer._copy_bbox(bbox))
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            plt.savefig(f.name, bbox_inches='tight')
            im_dst = ct.io.imread(f.name, alpha_mode="ignore")
        plt.close()

        return ct.image.crop_white_boarders(im_dst)

    def _redraw(self):
        for patch in self.visible_patches:
            patch.remove()
        self.visible_patches.clear()

        # Draw confirmed patches.
        for patch in self.confirm_patches:
            for axis in self.axes:
                cloned_patch = BBoxer._copy_bbox(patch)
                self.visible_patches.append(axis.add_patch(cloned_patch))

        # Draw current patch.
        if self.current_patch is not None:
            for axis in self.axes:
                cloned_patch = BBoxer._copy_bbox(self.current_patch)
                self.visible_patches.append(axis.add_patch(cloned_patch))

    def _save(self) -> None:
        """
        Save images with bounding boxes to disk. This function is called by the
        matplotlib event handler when the figure is closed.

        If self.confirm_patches is empty, then no bounding boxes will be drawn,
        but the images will still be saved.
        """
        dst_paths = [
            p.parent / f"bbox_{p.stem}{p.suffix}" for p in self.src_paths
        ]
        for src_path, dst_path in zip(self.src_paths, dst_paths):
            im_src = ct.io.imread(src_path)
            im_dst = BBoxer._overlay_bbox_on_image(im_src, self.confirm_patches)
            ct.io.imwrite(dst_path, im_dst)
            print(f"Saved {dst_path}")

    def _on_keypress(self, event):
        """
        Callback function for keypress.
        """
        print(f"Pressed: {event.key}")
        sys.stdout.flush()

        # Check if enter is pressed.
        if event.key == "enter":
            bbox = self.current_patch.get_bbox()
            self.confirm_patches.append(self.current_patch)
            self.current_patch = None
            self._redraw()
            print(f"BBox saved: {bbox}")

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

        # Rectangle selector on the first image.
        def on_selector(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            rect = plt.Rectangle(
                (min(x1, x2), min(y1, y2)),
                np.abs(x1 - x2),
                np.abs(y1 - y2),
                linewidth=self.line_width,
                edgecolor=self.edge_color,
                facecolor='none',
            )
            self.current_patch = rect
            self._redraw()

        selector = RectangleSelector(
            axis,
            on_selector,
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
        )

        # If not saved, the selector will go out-of-scope.
        self.axis_to_selector[axis] = selector

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
