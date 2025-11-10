from pathlib import Path
import camtools as ct
import os
import tempfile
import shutil


def instantiate_parser(parser):
    parser.add_argument(
        "input",
        type=Path,
        help="One or more image path(s). Supporting PNG and JPG.",
        nargs="+",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jpg", "png"],
        default=None,
        help="Output format (jpg or png). If not specified, no processing is done.",
    )
    parser.add_argument(
        "--inplace",
        "-i",
        action="store_true",
        help="If specified, the original file will be deleted, even for PNG->JPG case.",
    )
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=95,
        help="Quality of the output JPEG image, 1-100. Default is 95. Only works for JPG output format.",
    )
    parser.add_argument(
        "--skip_compression_ratio",
        type=float,
        default=1.0,
        help="Skip compression if the compression ratio is above this value (default: 1.0). "
        "Only applies to JPG->JPG compression to avoid recompressing already compressed images. "
        "Default is 1.0 to process all files (no skipping).",
    )
    return parser


def entry_point(_parser, args):
    """
    This is used by sub_parser.set_defaults(func=entry_point).
    The parser argument is not used.
    """
    # If no format is specified, do nothing.
    if args.format is None:
        print("No --format specified. No changes made to any files.")
        return 0

    # Validate quality flag only works for JPG format.
    if args.quality != 95 and args.format == "png":
        raise ValueError(
            "The --quality flag only works for JPG output format, not PNG."
        )

    # Collect all src_paths.
    src_paths = []
    src_paths += args.input
    src_paths = [
        path
        for path in src_paths
        if ct.sanity.is_jpg_path(path) or ct.sanity.is_png_path(path)
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
        # Determine output extension.
        if args.format == "jpg":
            output_ext = ".jpg"
        else:  # args.format == "png"
            output_ext = ".png"

        # Check if this is a format conversion or same format compression.
        src_is_jpg = ct.sanity.is_jpg_path(src_path)
        src_is_png = ct.sanity.is_png_path(src_path)

        if (src_is_jpg and args.format == "jpg") or (
            src_is_png and args.format == "png"
        ):
            # Same format: preserve original extension (e.g., .JPG, .jpeg).
            dst_path = src_path
        else:
            # Format conversion: use standard extension (.jpg or .png).
            dst_path = src_path.with_suffix(output_ext)

        if not args.inplace:
            dst_path = dst_path.parent / f"processed_{dst_path.name}"
        dst_paths.append(dst_path)

    # Print summary.
    pwd = Path.cwd().resolve().absolute()
    print("[Candidate files]")
    for src_path, dst_path in zip(src_paths, dst_paths):
        print(f"  - src: {Path(os.path.relpath(path=src_path, start=pwd))}")
        print(f"    dst: {Path(os.path.relpath(path=dst_path, start=pwd))}")
    print("[To be updated]")
    inplace_str = "src will be deleted" if args.inplace else "src will be kept"
    print(f"  - num_images: {len(src_paths)} files")
    print(f"  - format    : {args.format}")
    if args.format == "jpg":
        print(f"  - quality   : {args.quality}")
    print(f"  - inplace   : {inplace_str}")

    # Compress images.
    if not ct.util.query_yes_no("Proceed?", default=False):
        print("Aborted.")
        return 0
    stats = compress_images(
        src_paths=src_paths,
        dst_paths=dst_paths,
        output_format=args.format,
        quality=args.quality,
        delete_src=args.inplace,
        skip_compression_ratio=args.skip_compression_ratio,
    )

    # Print stats.
    src_sizes = [stat["src_size"] for stat in stats]
    dst_sizes = [stat["dst_size"] for stat in stats]
    src_total_size_mb = sum(src_sizes) / 1024 / 1024
    dst_total_size_mb = sum(dst_sizes) / 1024 / 1024
    compression_ratio = (
        dst_total_size_mb / src_total_size_mb if src_total_size_mb > 0 else 0
    )
    num_total = len(src_paths)
    num_direct_copy = len([stat for stat in stats if stat["is_direct_copy"]])
    num_compressed = num_total - num_direct_copy

    print("[Summary]")
    print(f"  - num_total        : {num_total}")
    print(f"  - num_direct_copy  : {num_direct_copy}")
    print(f"  - num_compressed   : {num_compressed}")
    print(f"  - src_total_size_mb: {src_total_size_mb:.2f} MB")
    print(f"  - dst_total_size_mb: {dst_total_size_mb:.2f} MB")
    print(f"  - compression_ratio: {compression_ratio:.2f}")


def compress_image_and_return_stat(
    src_path: Path,
    dst_path: Path,
    output_format: str,
    quality: int,
    delete_src: bool,
    skip_compression_ratio: float,
):
    """
    Compress image and return stats.

    Args:
        src_path: Path to image.
            - Only ".jpg" or ".png" is supported.
            - Directory will be created if it does not exist.
        dst_path: Path to image.
            - Only ".jpg" or ".png" is supported.
            - Directory will be created if it does not exist.
        output_format: Output format ("jpg" or "png").
        quality: Quality of the output JPEG image, 1-100. Default is 95.
            Only applicable for JPG output.
        delete_src: If True, the src_path will be deleted.
        skip_compression_ratio: Skip compression if the compression ratio is
            above this value. Only applies to JPG->JPG compression.

    Returns:
        stat: A dictionary of stats.
            {
                "src_path": Path to the source image.
                "dst_path": Path to the destination image.
                "src_size": Size of the source image in bytes.
                "dst_size": Size of the destination image in bytes.
                "compression_ratio": Compression ratio.
                "is_direct_copy": True if the image is skipped/copied directly.
            }

    Notes:
        - You should not use this to save a depth image (typically uint16).
        - Float image will get a range check to ensure it is in [0, 1].
        - PNG -> JPG: Alpha channel is removed (flattened to white background).
        - PNG -> PNG: Alpha channel is preserved.
    """
    stat = {}

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    stat["src_path"] = src_path
    stat["dst_path"] = dst_path

    # Read image with appropriate alpha handling.
    src_is_png = ct.sanity.is_png_path(src_path)
    src_is_jpg = ct.sanity.is_jpg_path(src_path)

    if src_is_png and output_format == "jpg":
        # PNG -> JPG: Remove alpha channel (flatten to white).
        im = ct.io.imread(src_path, alpha_mode="white")
    else:
        # PNG -> PNG: Keep alpha channel.
        # JPG -> JPG: No alpha channel.
        # JPG -> PNG: No alpha channel.
        im = ct.io.imread(src_path, alpha_mode="keep")

    src_size = src_path.stat().st_size

    # Determine file suffix for temporary file.
    temp_suffix = ".jpg" if output_format == "jpg" else ".png"

    # Write to a temporary file to get the file size.
    with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=True) as fp:
        if output_format == "jpg":
            ct.io.imwrite(fp.name, im, quality=quality)
        else:  # output_format == "png"
            ct.io.imwrite(fp.name, im)

        dst_size = Path(fp.name).stat().st_size
        compression_ratio = dst_size / src_size

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if we should skip compression for JPG->JPG.
        if (
            src_is_jpg
            and output_format == "jpg"
            and compression_ratio > skip_compression_ratio
        ):
            # The image is already compressed. Direct copy src_path to dst_path.
            # This avoids recompressing an image that is already compressed.
            # Keep the modification time, creation time, and permissions.
            if src_path != dst_path:
                shutil.copy2(src=src_path, dst=dst_path)
            stat["is_direct_copy"] = True
        else:
            # Copy from temp file to dst_path.
            fp.seek(0)
            with open(dst_path, "wb") as dst_fp:
                dst_fp.write(fp.read())
            stat["is_direct_copy"] = False

    # Recompute dst_size.
    dst_size = dst_path.stat().st_size
    compression_ratio = dst_size / src_size
    stat["src_size"] = src_size
    stat["dst_size"] = dst_size
    stat["compression_ratio"] = compression_ratio

    # Remove the source file if necessary.
    if delete_src and src_path != dst_path:
        src_path.unlink()

    return stat


def compress_images(
    src_paths,
    dst_paths,
    output_format: str,
    quality: int,
    delete_src: bool,
    skip_compression_ratio: float,
):
    """
    Compress images.

    Args:
        src_paths: List of source image paths.
        dst_paths: List of destination image paths.
        output_format: Output format ("jpg" or "png").
        quality: Quality of the output JPEG image, 1-100. Default is 95.
            Only applicable for JPG output.
        delete_src: If True, the src_path will be deleted.
        skip_compression_ratio: Skip compression if the compression ratio is
            above this value. Only applies to JPG->JPG compression.
    """
    stats = []
    for src_path, dst_path in zip(src_paths, dst_paths):
        stat = compress_image_and_return_stat(
            src_path=src_path,
            dst_path=dst_path,
            output_format=output_format,
            quality=quality,
            delete_src=delete_src,
            skip_compression_ratio=skip_compression_ratio,
        )
        stats.append(stat)
    return stats
