from pathlib import Path
import argparse
import numpy as np
import cv2
import camtools as ct
import os


def instantiate_parser(parser):
    parser.add_argument(
        "input",
        type=Path,
        help="One or more image path(s), or one directory. Supporting PNG and JPG.",
        nargs="+",
    )
    parser.add_argument(
        "--inplace",
        "-i",
        action="store_true",
        help="If specified, the original file will be deleted, even for PNG->JPG case.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="Quality of the output JPEG image, 1-100. Default is 80.",
    )
    return parser


def entry_point(parser, args):
    """
    This is used by sub_parser.set_defaults(func=entry_point).
    The parser argument is not used.
    """
    # Collect all src_paths.
    src_paths = []
    if len(args.input) == 1 and args.input[0].is_dir():
        src_dir = args.input[0]
        src_paths += list(src_dir.glob("**/*"))
    else:
        src_paths += args.input
    src_paths = [
        path for path in src_paths if ct.io.is_jpg_path(path) or ct.io.is_png_path(path)
    ]
    src_paths = [path.resolve().absolute() for path in src_paths]
    missing_paths = [path for path in src_paths if not path.is_file()]
    if not src_paths:
        raise ValueError("No image found.")
    if missing_paths:
        raise ValueError(f"Missing files: {missing_paths}")

    # Compute dst_paths.
    dst_paths = []
    for src_path in src_paths:
        if args.inplace:
            if ct.io.is_jpg_path(src_path):
                dst_paths.append(src_path)
            elif ct.io.is_png_path(src_path):
                dst_paths.append(src_path.with_suffix(".jpg"))
            else:
                raise ValueError(f"Unsupported image type: {src_path}")
        else:
            if ct.io.is_jpg_path(src_path):
                dst_paths.append(src_path.parent / f"compressed_{src_path.name}")
            elif ct.io.is_png_path(src_path):
                dst_paths.append(src_path.parent / f"compressed_{src_path.name}.jpg")
            else:
                raise ValueError(f"Unsupported image type: {src_path}")

    # Print summary.
    pwd = Path.cwd().resolve().absolute()
    print(f"[Files]")
    for src_path, dst_path in zip(src_paths, dst_paths):
        print(f"  - src: {Path(os.path.relpath(path=src_path, start=pwd))}")
        print(f"    dst: {Path(os.path.relpath(path=dst_path, start=pwd))}")
    print(f"[Configs]")
    inplace_str = "src will be deleted" if args.inplace else "src will be kept"
    print(f"  - inplace: {inplace_str}")
    print(f"  - quality: {args.quality}")

    # Confirm.
    if ct.utility.query_yes_no("Proceed?", default=False):
        print("Proceeding.")
    else:
        print("Aborted.")
        return 0
