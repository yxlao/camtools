from pathlib import Path
import camtools as ct
import os
import tempfile
import shutil


def instantiate_parser(parser):
    parser.add_argument(
        "input",
        type=Path,
        help="One or more image path(s), or one directory. Supporting PNG and JPG.",
        nargs="+",
    )
    parser.add_argument(
        "--png_only",
        action="store_true",
        help="If specified, only PNG files will be processed.",
    )
    parser.add_argument(
        "--flatten_alpha_channel",
        "-f",
        action="store_true",
        help="If specified, the alpha channel will be flattened to white.",
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
        help="Quality of the output JPEG image, 1-100. Default is 95.",
    )
    parser.add_argument(
        "--update_texts_in_dir",
        "-u",
        type=Path,
        help="Update text files (.txt, .md, .tex) in the directory to reflect the new image paths.",
    )
    parser.add_argument(
        "--min_jpg_compression_ratio",
        type=float,
        default=0.9,
        help="Minimum compression ratio for jpg->jpg compression. "
        "If the compression ratio is above this value, the image will not be compressed. "
        "This avoids compressing an image that is already compressed.",
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
    if args.png_only:
        src_paths = [path for path in src_paths if ct.sanity.is_png_path(path)]
    else:
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

    # Handle PNG file's alpha channel.
    src_paths_with_alpha = []
    png_paths = [src_path for src_path in src_paths if ct.sanity.is_png_path(src_path)]
    for src_path in png_paths:
        im = ct.io.imread(src_path, alpha_mode="keep")
        if im.shape[2] == 4:
            src_paths_with_alpha.append(src_path)
    if len(src_paths_with_alpha) > 0:
        # Skip PNG files with alpha channel if flatten is not specified.
        if not args.flatten_alpha_channel:
            src_paths = [
                src_path
                for src_path in src_paths
                if src_path not in src_paths_with_alpha
            ]

    # Compute dst_paths.
    dst_paths = []
    for src_path in src_paths:
        if ct.sanity.is_jpg_path(src_path):
            dst_path = src_path
        else:
            dst_path = src_path.with_suffix(".jpg")
        if not args.inplace:
            dst_path = dst_path.parent / f"compressed_{dst_path.name}"
        dst_paths.append(dst_path)

    # Check update_texts_in_dir.
    if args.update_texts_in_dir is not None:
        update_texts_in_dir = Path(args.update_texts_in_dir)
        if not update_texts_in_dir.is_dir():
            raise ValueError(
                f"update_texts_in_dir must be a directory, "
                f"but got {update_texts_in_dir}"
            )
        text_paths = get_all_text_paths(update_texts_in_dir)

    # Print summary.
    pwd = Path.cwd().resolve().absolute()
    print(f"[Candidate files]")
    for src_path, dst_path in zip(src_paths, dst_paths):
        print(f"  - src: {Path(os.path.relpath(path=src_path, start=pwd))}")
        print(f"    dst: {Path(os.path.relpath(path=dst_path, start=pwd))}")
    if len(src_paths_with_alpha) > 0:
        if args.flatten_alpha_channel:
            print("[PNG files with alpha channel (to be flattened)]")
            for src_path in src_paths_with_alpha:
                print(f"  - {Path(os.path.relpath(path=src_path, start=pwd))}")
            f"They will be skipped."
        else:
            print("[PNG files with alpha channel (to be skipped)]")
            for src_path in src_paths_with_alpha:
                print(f"  - {Path(os.path.relpath(path=src_path, start=pwd))}")
            print(
                f"{len(src_paths_with_alpha)} PNG files have alpha channel. "
                f"They will be skipped."
            )
    print(f"[To be updated]")
    inplace_str = "src will be deleted" if args.inplace else "src will be kept"
    if args.update_texts_in_dir is not None:
        print(f"  - num_images         : {len(src_paths)} files")
        print(f"  - quality            : {args.quality}")
        print(f"  - inplace            : {inplace_str}")
        print(
            f"  - update_texts_in_dir: {update_texts_in_dir} "
            f"({len(text_paths)} files)"
        )
    else:
        print(f"  - num_images: {len(src_paths)} files")
        print(f"  - quality   : {args.quality}")
        print(f"  - inplace   : {inplace_str}")

    # Compress images.
    if not ct.util.query_yes_no("Proceed?", default=False):
        print("Aborted.")
        return 0
    stats = compress_images(
        src_paths=src_paths,
        dst_paths=dst_paths,
        quality=args.quality,
        delete_src=args.inplace,
        min_jpg_compression_ratio=args.min_jpg_compression_ratio,
    )

    # Print stats.
    src_sizes = [stat["src_size"] for stat in stats]
    dst_sizes = [stat["dst_size"] for stat in stats]
    src_total_size_mb = sum(src_sizes) / 1024 / 1024
    dst_total_size_mb = sum(dst_sizes) / 1024 / 1024
    compression_ratio = dst_total_size_mb / src_total_size_mb
    num_total = len(src_paths)
    num_direct_copy = len([stat for stat in stats if stat["is_direct_copy"]])
    num_compressed = num_total - num_direct_copy

    print(f"[Summary]")
    print(f"  - num_total        : {num_total}")
    print(f"  - num_direct_copy  : {num_direct_copy}")
    print(f"  - num_compressed   : {num_compressed}")
    print(f"  - src_total_size_mb: {src_total_size_mb:.2f} MB")
    print(f"  - dst_total_size_mb: {dst_total_size_mb:.2f} MB")
    print(f"  - compression_ratio: {compression_ratio:.2f}")

    # Update text files.
    src_paths = [stat["src_path"] for stat in stats if not stat["is_direct_copy"]]
    dst_paths = [stat["dst_path"] for stat in stats if not stat["is_direct_copy"]]
    if num_compressed > 0 and update_texts_in_dir is not None:
        do_update_texts_in_dir(
            src_paths=src_paths,
            dst_paths=dst_paths,
            root_dir=update_texts_in_dir,
        )


def do_update_texts_in_dir(src_paths, dst_paths, root_dir):
    # Print update dict.
    root_dir = root_dir.resolve().absolute()
    replace_dict = {}
    for src_path, dst_path in zip(src_paths, dst_paths):
        src_path = src_path.relative_to(root_dir)
        dst_path = dst_path.relative_to(root_dir)
        replace_dict[str(src_path)] = str(dst_path)
    keys = sorted(replace_dict.keys())
    print(f"[With replace dict of {len(keys)} patterns]")
    for key in keys:
        print(f"  - {key} -> {replace_dict[key]}")

    # Collect all text paths.
    text_paths = get_all_text_paths(root_dir)
    print(f"[Updating {len(text_paths)} text files]")
    for text_path in text_paths:
        print(f"  - {text_path}")

    prompt = "Continue?"
    if ct.util.query_yes_no(prompt, default=False):
        pass
    else:
        print("Aborted.")
        return 0

    same_paths = []
    updated_paths = []
    for text_path in text_paths:
        before_text = text_path.read_text()
        replace_strings_in_file(path=text_path, replace_dict=replace_dict)
        after_text = text_path.read_text()
        if before_text == after_text:
            same_paths.append(text_path)
        else:
            updated_paths.append(text_path)

    print(f"[Summary]")
    print(f"  - num_same   : {len(same_paths)}")
    print(f"  - num_updated: {len(updated_paths)}")
    for text_path in updated_paths:
        print(f"    - {text_path}")


def compress_image_and_return_stat(
    src_path: Path,
    dst_path: Path,
    quality: int,
    delete_src: bool,
    min_jpg_compression_ratio: float,
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
        quality: Quality of the output JPEG image, 1-100. Default is 95.
        delete_src: If True, the src_path will be deleted.
        min_jpg_compression_ratio: Minimum compression ratio for jpg->jpg
            compression. If the compression ratio is above this value, the image
            will not be compressed. This avoids compressing an image that is
            already compressed.

    Returns:
        stat: A dictionary of stats.
            {
                "src_path": Path to the source image.
                "dst_path": Path to the destination image.
                "src_size": Size of the source image in bytes.
                "dst_size": Size of the destination image in bytes.
                "compression_ratio": Compression ratio.
                "is_direct_copy": True if the image is already compressed.
            }

    Notes:
        - You should not use this to save a depth image (typically uint16).
        - Float image will get a range check to ensure it is in [0, 1].
    """
    stat = {}

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not ct.sanity.is_jpg_path(dst_path):
        raise ValueError(f"dst_path must be a JPG file: {dst_path}")
    stat["src_path"] = src_path
    stat["dst_path"] = dst_path

    # Read.
    im = ct.io.imread(src_path)
    src_size = src_path.stat().st_size

    # Write to a temporary file to get the file size.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as fp:
        ct.io.imwrite(fp.name, im, quality=quality)
        dst_size = Path(fp.name).stat().st_size
        compression_ratio = dst_size / src_size

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            ct.sanity.is_jpg_path(src_path)
            and compression_ratio > min_jpg_compression_ratio
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


def get_all_text_paths(root_dir):
    def is_text_file(path):
        return path.is_file() and path.suffix in [".txt", ".md", ".tex"]

    root_dir = Path(root_dir)
    text_paths = list(root_dir.glob("**/*"))
    text_paths = [text_path for text_path in text_paths if is_text_file(text_path)]
    return text_paths


def replace_strings_in_file(path, replace_dict):
    """
    Replace strings in a file.

    Args:
        path: Path to the file.
        replace_dict: A dictionary of strings to be replaced.
            - Keys are the strings to be replaced.
            - Values are the strings to replace with.
    """
    with path.open() as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    for i, line in enumerate(lines):
        for src_str, dst_str in replace_dict.items():
            lines[i] = lines[i].replace(src_str, dst_str)
    with path.open("w") as f:
        for line in lines:
            f.write(f"{line}\n")


def compress_images(
    src_paths,
    dst_paths,
    quality: int,
    delete_src: bool,
    min_jpg_compression_ratio: float,
):
    """
    Compress images (PNGs will be converted to JPGs)

    Args:
        src_paths: List of source image paths.
        dst_paths: List of destination image paths.
        quality: Quality of the output JPEG image, 1-100. Default is 95.
        delete_src: If True, the src_path will be deleted.
        min_jpg_compression_ratio: Minimum compression ratio for jpg->jpg
            compression. If the compression ratio is above this value, the image
            will not be compressed. This avoids compressing an image that is
            already compressed.
    """
    stats = []
    for src_path, dst_path in zip(src_paths, dst_paths):
        stat = compress_image_and_return_stat(
            src_path=src_path,
            dst_path=dst_path,
            quality=quality,
            delete_src=delete_src,
            min_jpg_compression_ratio=min_jpg_compression_ratio,
        )
        stats.append(stat)
    return stats
