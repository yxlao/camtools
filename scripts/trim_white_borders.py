"""
Trim white borders of an image.

Example usage:
```bash
python $HOME/research/camtools/scripts/trim_white_borders.py *.png --pad_pixel 10 --same_shape --skip_trimmed
```

TODO: make this an executable, e.g.:
```bash
ct trim_white_borders xxx
ct --help
ct trim_white_borders --help
```
"""

from pathlib import Path
import argparse
import numpy as np
import camtools as ct


def trim_white_boarders(im, padding):
    """
    Trim white borders of an image.

    Args:
        im: (H, W, 3) image, float32.
        padding: int, padding size on each edge.
    """
    if not im.dtype == np.float32:
        raise ValueError(f"im.dtype == {im.dtype} != np.float32")
    if not im.ndim == 3:
        raise ValueError(f"im.shape must be (H, W, 3), but got {im.shape}")
    if not im.shape[2] == 3:
        raise ValueError(f"im.shape must be (H, W, 3), but got {im.shape}")

    h, w, _ = im.shape

    # Find the number of white pixels on each edge.
    num_u = 0
    num_d = 0
    num_l = 0
    num_r = 0

    for u in range(h):
        if np.allclose(im[u, :, :], 1.0):
            num_u += 1
        else:
            break
    for d in range(h):
        if np.allclose(im[h - d - 1, :, :], 1.0):
            num_d += 1
        else:
            break
    for l in range(w):
        if np.allclose(im[:, l, :], 1.0):
            num_l += 1
        else:
            break
    for r in range(w):
        if np.allclose(im[:, w - r - 1, :], 1.0):
            num_r += 1
        else:
            break

    # Sanity checks.
    if num_u + num_d >= h:
        raise ValueError(f"num_u + num_d >= h: {num_u} + {num_d} >= {h}")

    if num_l + num_r >= w:
        raise ValueError(f"num_l + num_r >= w: {num_l} + {num_r} >= {w}")

    # Crop.
    im = im[num_u:h - num_d, num_l:w - num_r, :]

    # Pad white color by padding.
    im = np.pad(
        im,
        ((padding, padding), (padding, padding), (0, 0)),
        constant_values=1,
    )

    return im


def trim_single_image(in_path, out_path, pad_pixel, pad_ratio):
    if not in_path.exists():
        raise FileNotFoundError(f"Input file {in_path} does not exist.")

    # Read.
    im = ct.io.imread(in_path)

    # Compute padding.
    if pad_pixel != 0:
        padding = pad_pixel
    else:
        h, w, _ = im.shape
        padding = int(max(h, w) * pad_ratio)

    # Convert.
    im_trimmed = trim_white_boarders(im, padding=padding)

    # Save.
    out_dir = out_path.parent
    if not out_dir.exists():
        print(f"Creating directory {out_dir}")
        out_dir.mkdir(parents=True)
    ct.io.imwrite(out_path, im_trimmed)

    # Print summary.
    print(f"[Trim]")
    print(f"  - Input       : {in_path}")
    print(f"  - Output      : {out_path}")
    print(f"  - Padding     : {padding}")
    print(f"  - Input shape : {im.shape}")
    print(f"  - Output shape: {im_trimmed.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        help="Input image path.",
        nargs="+",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path.",
        default=None,
    )
    parser.add_argument(
        "--pad_pixel",
        type=int,
        default=0,
        help="Padding size on each edge. When specified, pad_ratio is ignored.",
    )
    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.0,
        help="Padding size on each edge, dividing the longer edge pixel count "
        "of the image. When pad_pixel is specified, this option is ignored.",
    )
    parser.add_argument(
        "--same_shape",
        action="store_true",
        help="If specified, all output images will be padded to the same shape. "
        "This is useful for plotting them in a grid.",
    )
    parser.add_argument(
        "--skip_trimmed",
        action="store_true",
        help="If specified, skip trimming images that have already been "
        "trimmed. A trimmed image is identified by the existence of a "
        "file with the same name as the input file, but with the suffix "
        "'_trimmed'.",
    )

    args = parser.parse_args()

    if args.pad_pixel < 0:
        raise ValueError(
            f"pad_pixel must be non-negative, but got {args.pad_pixel}")
    if args.pad_ratio < 0:
        raise ValueError(
            f"pad_ratio must be non-negative, but got {args.pad_ratio}")

    if isinstance(args.input, list):
        in_paths = args.input
        if args.output is not None:
            raise ValueError(
                "When input is a list, output must not be specified.")

        if args.skip_trimmed:
            out_paths = [
                in_path.parent / f"{in_path.stem}_trimmed{in_path.suffix}"
                for in_path in in_paths
            ]
            new_in_paths = []
            for in_path in in_paths:
                if in_path in out_paths and in_path.stem.endswith("_trimmed"):
                    src_in_path = in_path.parent / f"{in_path.stem[:-8]}{in_path.suffix}"
                    if src_in_path.exists():
                        print(
                            f"Skipping {in_path}, as it is trimmed from {src_in_path}"
                        )
                        continue
                new_in_paths.append(in_path)
            in_paths = new_in_paths

        # Trim multiple images.
        out_paths = []
        for in_path in in_paths:
            suffix = in_path.suffix
            out_path = in_path.parent / f"{in_path.stem}_trimmed{suffix}"
            out_paths.append(out_path)

            trim_single_image(
                in_path=in_path,
                out_path=out_path,
                pad_pixel=args.pad_pixel,
                pad_ratio=args.pad_ratio,
            )

        # Pad images to the same shape.
        if args.same_shape:
            max_h = 0
            max_w = 0
            for out_path in out_paths:
                im = ct.io.imread(out_path)
                h, w, _ = im.shape
                max_h = max(max_h, h)
                max_w = max(max_w, w)
            for out_path in out_paths:
                im = ct.io.imread(out_path)
                h, w, _ = im.shape
                if h < max_h or w < max_w:
                    h_pad = max_h - h
                    w_pad = max_w - w
                    h_pad_top = h_pad // 2
                    h_pad_bottom = h_pad - h_pad_top
                    w_pad_left = w_pad // 2
                    w_pad_right = w_pad - w_pad_left
                    im_padded = np.pad(
                        im,
                        ((h_pad_top, h_pad_bottom), (w_pad_left, w_pad_right),
                         (0, 0)),
                        constant_values=1,
                    )
                else:
                    im_padded = im

                ct.io.imwrite(out_path, im_padded)
                print(f"[Pad to same]")
                print(f"  - Path        : {out_path}")
                print(f"  - Input shape : {im.shape}")
                print(f"  - Output shape: {im_padded.shape}")

    else:
        in_path = Path(args.input)
        suffix = in_path.suffix
        if args.output is None:
            out_path = in_path.parent / f"{in_path.stem}_trimmed{suffix}"
        else:
            out_path = Path(args.output)
        trim_single_image(
            in_path=in_path,
            out_path=out_path,
            pad_pixel=args.pad_pixel,
            pad_ratio=args.pad_ratio,
        )


if __name__ == "__main__":
    main()
