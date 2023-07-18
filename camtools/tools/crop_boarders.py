"""
Crop white borders of an image.

Example usage:

```python
ct crop-boarders *.png --pad_pixel 10 --skip_cropped --same_crop
```
"""

from pathlib import Path
import argparse
import numpy as np
import camtools as ct


def instantiate_parser(parser):
    parser.add_argument(
        "input",
        type=Path,
        help="Input image paths.",
        nargs="+",
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
        help="All output images will be padded to the same shape. This is "
        "useful for plotting images in a grid.",
    )
    parser.add_argument(
        "--same_crop",
        action="store_true",
        help="All images will be stacked and cropped together. This ensures "
        "the images are cropped with the exact amount from all edges. This "
        "requires all input images are of the same shape. This option is "
        "useful when the contents in images are aligned.",
    )
    parser.add_argument(
        "--skip_cropped",
        action="store_true",
        help="If specified, skip cropping images that have already been "
        "cropped. A cropped image is identified by the existence of a "
        "file with the same name as the input file, but with the prefix "
        "'cropped_'.",
    )
    parser.add_argument(
        "--inplace",
        "-i",
        action="store_true",
        help="If specified, overwrite the input file with the cropped image.",
    )
    return parser


def entry_point(parser, args):
    """
    This is used by sub_parser.set_defaults(func=entry_point).
    The parser argument is not used.
    """
    if args.pad_pixel < 0:
        raise ValueError(f"pad_pixel must be non-negative, but got {args.pad_pixel}")
    if args.pad_ratio < 0:
        raise ValueError(f"pad_ratio must be non-negative, but got {args.pad_ratio}")

    # Determine src and dst paths.
    if isinstance(args.input, list):
        src_paths = args.input
    else:
        src_paths = [args.input]
    for src_path in src_paths:
        if not src_path.is_file():
            raise FileNotFoundError(f"Input file {src_path} does not exist.")
    if args.inplace:
        if args.skip_cropped:
            raise ValueError(
                "Cannot specify --skip_cropped when --inplace is specified."
            )
        dst_paths = src_paths
    else:
        if args.skip_cropped:
            dst_paths = [
                src_path.parent / f"cropped_{src_path.name}" for src_path in src_paths
            ]
            skipped_src_paths = [p for p in src_paths if p in dst_paths]
            src_paths = [p for p in src_paths if p not in dst_paths]
            if len(skipped_src_paths) > 0:
                print("[Skipped]")
                for src_path in skipped_src_paths:
                    print(f"  - {src_path}")
        dst_paths = [
            src_path.parent / f"cropped_{src_path.name}" for src_path in src_paths
        ]

    # Read.
    src_ims = [ct.io.imread(src_path) for src_path in src_paths]
    for src_im in src_ims:
        if not src_im.dtype == np.float32:
            raise ValueError(f"Input image {src_path} must be of dtype float32.")
        if not src_im.ndim == 3 or not src_im.shape[2] == 3:
            raise ValueError(f"Input image {src_path} must be of shape (H, W, 3).")
    num_ims = len(src_ims)

    # Compute.
    if args.same_crop:
        # Check all images are of the same shape.
        shapes = [im.shape for im in src_ims]
        if not all([s == shapes[0] for s in shapes]):
            raise ValueError(
                "All images must be of the same shape when --same_crop is " "specified."
            )

        # Stack images.
        src_ims_stacked = np.concatenate(src_ims, axis=2)

        # Compute cropping boarders.
        crop_u, crop_d, crop_l, crop_r = ct.image.compute_cropping(src_ims_stacked)
        croppings = [(crop_u, crop_d, crop_l, crop_r)] * num_ims

        # Compute padding.
        if args.pad_pixel != 0:
            padding = args.pad_pixel
        else:
            h, w, _ = src_ims_stacked.shape
            padding = int(max(h, w) * args.pad_ratio)
        paddings = [(padding, padding, padding, padding)] * num_ims
    else:
        # Compute cropping boarders.
        croppings = [ct.image.compute_cropping(src_im) for src_im in src_ims]

        # Compute paddings.
        if args.pad_pixel != 0:
            padding = args.pad_pixel
            paddings = [(padding, padding, padding, padding)] * num_ims
        else:
            paddings = []
            for src_im in src_ims:
                h, w, _ = src_im.shape
                padding = int(max(h, w) * args.pad_ratio)
                paddings.append((padding, padding, padding, padding))

        # If same_shape is specified, pad all images to the same shape.
        # Distribute the padding evenly among top/down, left/right.
        if args.same_shape:
            dst_shapes = ct.image.get_post_croppings_paddings_shapes(
                src_shapes=[im.shape for im in src_ims],
                croppings=croppings,
                paddings=paddings,
            )
            dst_shapes = np.array(dst_shapes)
            max_h = dst_shapes[:, 0].max()
            max_w = dst_shapes[:, 1].max()
            extra_paddings = []
            for dst_shape in dst_shapes:
                h, w, _ = dst_shape
                dh = max_h - h
                dw = max_w - w
                extra_paddings.append(
                    (
                        dh // 2,
                        dh - dh // 2,
                        dw // 2,
                        dw - dw // 2,
                    )
                )
            for i in range(num_ims):
                paddings[i] = tuple(np.array(paddings[i]) + np.array(extra_paddings[i]))

    # Apply.
    dst_ims = ct.image.apply_croppings_paddings(
        src_ims=src_ims,
        croppings=croppings,
        paddings=paddings,
    )

    # Save.
    for (
        src_path,
        dst_path,
        src_im,
        dst_im,
        cropping,
        padding,
    ) in zip(
        src_paths,
        dst_paths,
        src_ims,
        dst_ims,
        croppings,
        paddings,
    ):
        out_dir = dst_path.parent
        if not out_dir.exists():
            print(f"Creating directory {out_dir}")
            out_dir.mkdir(parents=True)
        ct.io.imwrite(dst_path, dst_im)

        print("[Cropped]")
        print(f"  - Input       : {src_path}")
        print(f"  - Output      : {dst_path}")
        print(f"  - Cropping    : {cropping}")
        print(f"  - Padding     : {padding}")
        print(f"  - Input shape : {src_im.shape}")
        print(f"  - Output shape: {dst_im.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser = instantiate_parser(parser)
    args = parser.parse_args()
    entry_point(parser, args)


if __name__ == "__main__":
    main()
