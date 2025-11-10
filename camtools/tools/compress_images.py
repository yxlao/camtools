from pathlib import Path
import camtools as ct
import os
import tempfile
import shutil
from rich.console import Console
from rich.table import Table
from rich import box
from tqdm import tqdm


def instantiate_parser(parser):
    parser.description = (
        "Compress and convert images between PNG and JPG formats.\n\n"
        "Default behavior:\n"
        "  - No --format specified: No changes made to any files.\n\n"
        "Format conversions:\n"
        "  - PNG -> JPG: Alpha channel is removed (flattened to white background).\n"
        "  - PNG -> PNG: Alpha channel is preserved.\n"
        "  - JPG -> JPG or PNG -> PNG: Original extension is preserved (e.g., .JPG, .jpeg).\n"
        "  - Format conversion (PNG -> JPG or JPG -> PNG): Standard extension is used (.jpg or .png).\n\n"
        "Output file naming:\n"
        "  - Without --inplace: 'processed_<filename>' is created, original file is kept.\n"
        "  - With --inplace: Original file is replaced/deleted.\n\n"
        "Examples:\n"
        "  # Convert PNG to JPG with quality 90\n"
        "  ct compress-images image.png --format jpg --quality 90\n\n"
        "  # Compress JPG inplace, skip if compression ratio > 0.9\n"
        "  ct compress-images image.jpg --format jpg --quality 90 --skip_compression_ratio 0.9 --inplace\n\n"
        "  # Convert recursively all PNG to JPG\n"
        "  ct compress-images **/*.png --format jpg --quality 90\n"
    )
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
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip user confirmation and proceed automatically.",
    )
    return parser


def get_operation_string(src_is_png, output_format, quality):
    """
    Get a short string describing the operation.
    """
    if src_is_png and output_format == "jpg":
        return f"PNG→JPG Q{quality}"
    elif not src_is_png and output_format == "png":
        return "JPG→PNG"
    elif output_format == "jpg":
        return f"JPG→JPG Q{quality}"
    else:
        return "PNG→PNG"


def print_file_table(console, file_ops, stats=None, show_results=False):
    """
    Print a table showing file operations.
    If stats is None, print expected operations (before processing).
    If stats is provided, print results (after processing).
    """
    if show_results:
        console.print("\n[bold cyan]Compression Results[/bold cyan]\n")
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan", no_wrap=False, overflow="fold")
        table.add_column("Operation", justify="center", style="yellow")
        table.add_column("Input", justify="right", style="blue")
        table.add_column("Output", justify="right", style="blue")
        table.add_column("Ratio", justify="right", style="magenta")
        table.add_column("Saved", justify="right", style="green")
        table.add_column("Status", justify="center")

        for op, stat in zip(file_ops, stats):
            src_size_mb = stat["src_size"] / 1024 / 1024
            dst_size_mb = stat["dst_size"] / 1024 / 1024
            ratio = stat["compression_ratio"]
            savings_mb = src_size_mb - dst_size_mb
            savings_pct = (1 - ratio) * 100

            if stat["is_direct_copy"]:
                status = "[yellow]Skipped[/yellow]"
                saved_display = "-"
            elif savings_mb > 0:
                status = "[green]✓ Compressed[/green]"
                saved_display = f"{savings_mb:.2f} MB\n({savings_pct:.1f}%)"
            else:
                status = "[red]⚠ Larger[/red]"
                saved_display = f"{-savings_mb:.2f} MB\n({-savings_pct:.1f}%)"

            table.add_row(
                op["src_rel"],
                op["operation"],
                f"{src_size_mb:.2f} MB",
                f"{dst_size_mb:.2f} MB",
                f"{ratio:.3f}",
                saved_display,
                status,
            )
    else:
        console.print("\n[bold cyan]Expected File Operations[/bold cyan]\n")
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Input File", style="cyan", no_wrap=False, overflow="fold")
        table.add_column("Output", style="green", no_wrap=False, overflow="fold")
        table.add_column("Operation", justify="center", style="yellow")
        table.add_column("Size", justify="right", style="blue")
        table.add_column("Action", justify="center")

        for op in file_ops:
            table.add_row(
                op["src_rel"],
                op["dst_rel"],
                op["operation"],
                f"{op['src_size_mb']:.2f} MB",
                op["action"],
            )

    console.print(table)


def entry_point(_parser, args):
    """
    Main entry point for compress-images command.
    """
    # No format = do nothing.
    if args.format is None:
        print("No --format specified. No changes made to any files.")
        return 0

    # Quality only works for JPG.
    if args.quality != 95 and args.format == "png":
        raise ValueError(
            "The --quality flag only works for JPG output format, not PNG."
        )

    # Collect and validate source paths.
    src_paths = [
        p for p in args.input if ct.sanity.is_jpg_path(p) or ct.sanity.is_png_path(p)
    ]
    src_paths = [p.resolve().absolute() for p in src_paths]
    missing = [p for p in src_paths if not p.is_file()]

    if not src_paths:
        raise ValueError("No image found.")
    if missing:
        raise ValueError(f"Missing files: {missing}")

    # Prepare file operations.
    pwd = Path.cwd().resolve().absolute()
    console = Console()
    file_ops = []
    dst_paths = []

    for src_path in src_paths:
        src_is_png = ct.sanity.is_png_path(src_path)
        src_is_jpg = ct.sanity.is_jpg_path(src_path)

        # Determine destination path.
        if (src_is_jpg and args.format == "jpg") or (
            src_is_png and args.format == "png"
        ):
            # Same format: preserve extension.
            dst_path = src_path
        else:
            # Format conversion: use standard extension.
            dst_path = src_path.with_suffix(".jpg" if args.format == "jpg" else ".png")

        if not args.inplace:
            dst_path = dst_path.parent / f"processed_{dst_path.name}"

        dst_paths.append(dst_path)

        # Prepare display info.
        src_rel = str(Path(os.path.relpath(src_path, pwd)))
        dst_rel = str(Path(os.path.relpath(dst_path, pwd)))
        operation = get_operation_string(src_is_png, args.format, args.quality)

        if args.inplace:
            action = (
                "[red]Delete[/red]"
                if src_path != dst_path
                else "[yellow]Replace[/yellow]"
            )
        else:
            action = "[green]Keep[/green]"

        file_ops.append(
            {
                "src_path": src_path,
                "dst_path": dst_path,
                "src_rel": src_rel,
                "dst_rel": dst_rel,
                "operation": operation,
                "src_size_mb": src_path.stat().st_size / 1024 / 1024,
                "action": action,
            }
        )

    # Show expected operations.
    print_file_table(console, file_ops)

    # Print summary.
    summary = f"[bold]{len(src_paths)}[/bold] file(s) | Format: [bold]{args.format.upper()}[/bold]"
    if args.format == "jpg":
        summary += f" | Quality: [bold]{args.quality}[/bold] | Skip ratio: [bold]{args.skip_compression_ratio}[/bold]"
    summary += f" | Inplace: [bold]{'Yes' if args.inplace else 'No'}[/bold]"
    console.print(f"\n{summary}\n")

    # Ask for confirmation.
    if not args.yes:
        if not ct.util.query_yes_no("Proceed?", default=False):
            console.print("[yellow]Aborted.[/yellow]")
            return 0
    else:
        console.print("[green]--yes specified, proceeding automatically...[/green]\n")

    # Process images.
    console.print("[bold cyan]Processing Files...[/bold cyan]\n")
    stats = []
    with tqdm(total=len(src_paths), desc="Processing", unit="file") as pbar:
        for src_path, dst_path in zip(src_paths, dst_paths):
            stat = compress_image(
                src_path,
                dst_path,
                args.format,
                args.quality,
                args.inplace,
                args.skip_compression_ratio,
            )
            stats.append(stat)
            pbar.update(1)

    # Show results.
    print_file_table(console, file_ops, stats, show_results=True)

    # Print overall summary.
    src_total = sum(s["src_size"] for s in stats) / 1024 / 1024
    dst_total = sum(s["dst_size"] for s in stats) / 1024 / 1024
    total_saved = src_total - dst_total
    total_ratio = dst_total / src_total if src_total > 0 else 0
    num_skipped = sum(1 for s in stats if s["is_direct_copy"])

    console.print("\n[bold cyan]Overall Summary[/bold cyan]\n")
    summary_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Total files", f"{len(stats)}")
    summary_table.add_row("Processed", f"[green]{len(stats) - num_skipped}[/green]")
    summary_table.add_row("Skipped", f"[yellow]{num_skipped}[/yellow]")
    summary_table.add_row("Input size", f"{src_total:.2f} MB")
    summary_table.add_row("Output size", f"{dst_total:.2f} MB")

    if total_saved > 0:
        summary_table.add_row(
            "Saved", f"[green]{total_saved:.2f} MB ({(1-total_ratio)*100:.1f}%)[/green]"
        )
    else:
        summary_table.add_row(
            "Change", f"[red]+{-total_saved:.2f} MB ({(total_ratio-1)*100:.1f}%)[/red]"
        )

    summary_table.add_row("Overall ratio", f"{total_ratio:.3f}")

    console.print(summary_table)
    console.print()


def compress_image(
    src_path, dst_path, output_format, quality, delete_src, skip_compression_ratio
):
    """
    Compress a single image and return stats.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    src_is_png = ct.sanity.is_png_path(src_path)
    src_is_jpg = ct.sanity.is_jpg_path(src_path)

    # Read image.
    if src_is_png and output_format == "jpg":
        im = ct.io.imread(src_path, alpha_mode="white")  # Flatten alpha to white.
    else:
        im = ct.io.imread(src_path, alpha_mode="keep")  # Keep alpha for PNG->PNG.

    src_size = src_path.stat().st_size

    # Write to temp file to check compression ratio.
    suffix = ".jpg" if output_format == "jpg" else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as fp:
        if output_format == "jpg":
            ct.io.imwrite(fp.name, im, quality=quality)
        else:
            ct.io.imwrite(fp.name, im)

        temp_size = Path(fp.name).stat().st_size
        ratio = temp_size / src_size

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip compression if ratio is too high for JPG->JPG.
        if src_is_jpg and output_format == "jpg" and ratio > skip_compression_ratio:
            if src_path != dst_path:
                shutil.copy2(src_path, dst_path)
            is_direct_copy = True
        else:
            fp.seek(0)
            dst_path.write_bytes(fp.read())
            is_direct_copy = False

    # Get final size.
    dst_size = dst_path.stat().st_size

    # Delete source if needed.
    if delete_src and src_path != dst_path:
        src_path.unlink()

    return {
        "src_path": src_path,
        "dst_path": dst_path,
        "src_size": src_size,
        "dst_size": dst_size,
        "compression_ratio": dst_size / src_size,
        "is_direct_copy": is_direct_copy,
    }
