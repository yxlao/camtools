import subprocess
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest


def create_test_image_png(path: Path, with_alpha: bool = False):
    """
    Create a test PNG image using OpenCV.
    """
    if with_alpha:
        # Create RGBA image
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        # OpenCV uses BGRA
        cv2.imwrite(str(path), img)
    else:
        # Create RGB image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)


def create_test_image_jpg(path: Path, quality: int = 95):
    """
    Create a test JPG image using OpenCV.
    """
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _run_command(cmd: str, input_text: str = None):
    """
    Run cmd in terminal and return result.

    Args:
        cmd (str): Command to run in terminal.
        input_text (str): Text to send to stdin (e.g., 'y\n' or 'n\n').

    Returns:
        subprocess.CompletedProcess
    """
    cmd_tokens = cmd.split()
    proc = subprocess.run(
        cmd_tokens,
        capture_output=True,
        check=False,
        input=input_text.encode() if input_text else None,
    )
    return proc


def test_compress_images_help():
    """
    Test that the help command works.
    """
    proc = _run_command("ct compress-images --help")
    assert proc.returncode == 0
    assert b"usage: ct compress-images" in proc.stdout


def test_compress_images_no_format():
    """
    Test default behavior (no --format flag): should do nothing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images
        png_path = tmpdir / "image.png"
        jpg_path = tmpdir / "image.jpg"
        create_test_image_png(png_path)
        create_test_image_jpg(jpg_path)

        # Run command without --format
        cmd = f"ct compress-images {png_path} {jpg_path}"
        proc = _run_command(cmd)

        # Should succeed and report no changes
        assert proc.returncode == 0
        assert b"No --format specified" in proc.stdout

        # Original files should still exist
        assert png_path.exists()
        assert jpg_path.exists()

        # No new files should be created
        assert len(list(tmpdir.glob("*"))) == 2


def test_compress_images_png_to_jpg():
    """
    Test PNG to JPG conversion (not inplace).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test PNG
        src_path = tmpdir / "image.png"
        create_test_image_png(src_path)

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format jpg --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Original file should still exist
        assert src_path.exists()

        # Output file should be created without prefix (format conversion)
        dst_path = tmpdir / "image.jpg"
        assert dst_path.exists()

        # Check file is valid JPG
        img = cv2.imread(str(dst_path))
        assert img is not None
        assert img.shape == (100, 100, 3)


def test_compress_images_png_to_jpg_inplace():
    """
    Test PNG to JPG conversion (inplace).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test PNG
        src_path = tmpdir / "image.png"
        create_test_image_png(src_path)

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format jpg --inplace --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Original PNG should be deleted
        assert not src_path.exists()

        # Output JPG should be created
        dst_path = tmpdir / "image.jpg"
        assert dst_path.exists()

        # Check file is valid JPG
        img = cv2.imread(str(dst_path))
        assert img is not None
        assert img.shape == (100, 100, 3)


def test_compress_images_jpeg_to_png():
    """
    Test JPG to PNG conversion (not inplace).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test JPG with .jpeg extension
        src_path = tmpdir / "image.jpeg"
        create_test_image_jpg(src_path)

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format png --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Original file should still exist
        assert src_path.exists()

        # Output file should be created without prefix (format conversion)
        dst_path = tmpdir / "image.png"
        assert dst_path.exists()

        # Check file is valid PNG
        img = cv2.imread(str(dst_path))
        assert img is not None
        assert img.shape == (100, 100, 3)


def test_compress_images_jpg_to_jpg():
    """
    Test JPG to JPG compression (not inplace).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test JPG
        src_path = tmpdir / "image.jpg"
        create_test_image_jpg(src_path)

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format jpg --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Original file should still exist
        assert src_path.exists()

        # Output file should be created with processed_ prefix
        dst_path = tmpdir / "processed_image.jpg"
        assert dst_path.exists()

        # Check file is valid JPG
        img = cv2.imread(str(dst_path))
        assert img is not None
        assert img.shape == (100, 100, 3)


def test_compress_images_jpg_to_jpg_inplace():
    """
    Test JPG to JPG compression (inplace, preserves extension).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test JPG with .jpeg extension
        src_path = tmpdir / "image.jpeg"
        create_test_image_jpg(src_path)
        orig_size = src_path.stat().st_size

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format jpg --inplace --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Original file should still exist (inplace modification)
        assert src_path.exists()

        # Extension should be preserved (.jpeg)
        assert src_path.suffix == ".jpeg"

        # File should still be valid
        img = cv2.imread(str(src_path))
        assert img is not None
        assert img.shape == (100, 100, 3)


def test_compress_images_quality():
    """
    Test quality parameter for JPG output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test PNG
        src_path = tmpdir / "image.png"
        create_test_image_png(src_path)

        # Run command with quality 80 and --yes flag
        cmd = f"ct compress-images {src_path} --format jpg --quality 80 --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Output file should be created without prefix (format conversion)
        dst_path = tmpdir / "image.jpg"
        assert dst_path.exists()


def test_compress_images_quality_with_png_output_fails():
    """
    Test that quality parameter raises error for PNG output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test JPG
        src_path = tmpdir / "image.jpg"
        create_test_image_jpg(src_path)

        # Run command with quality for PNG output (should fail)
        cmd = f"ct compress-images {src_path} --format png --quality 80"
        proc = _run_command(cmd)

        # Should fail
        assert proc.returncode != 0
        assert b"quality" in proc.stderr.lower() or b"quality" in proc.stdout.lower()


def test_compress_images_skip_compression_ratio():
    """
    Test skip_compression_ratio parameter.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test JPG with high quality (hard to compress further)
        src_path = tmpdir / "image.jpg"
        create_test_image_jpg(src_path, quality=95)

        # Run command with skip_compression_ratio=0.5 (very low, should skip most images) and --yes flag
        cmd = f"ct compress-images {src_path} --format jpg --quality 80 --skip_compression_ratio 0.5 --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Check that the command completed (result is in summary, might be Skipped 1 or Processed 1)
        assert b"Overall Summary" in proc.stdout


def test_compress_images_multiple_files():
    """
    Test processing multiple files at once.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple test images
        png1_path = tmpdir / "image1.png"
        png2_path = tmpdir / "image2.png"
        jpg1_path = tmpdir / "image1.jpg"
        create_test_image_png(png1_path)
        create_test_image_png(png2_path)
        create_test_image_jpg(jpg1_path)

        # Run command on all files with --yes flag
        cmd = f"ct compress-images {png1_path} {png2_path} {jpg1_path} --format jpg --quality 90 --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Check output files:
        # PNG->JPG conversions: no prefix (different extensions)
        assert (tmpdir / "image1.jpg").exists()  # From png1
        assert (tmpdir / "image2.jpg").exists()  # From png2
        # JPG->JPG compression: has prefix (same format)
        assert (tmpdir / "processed_image1.jpg").exists()  # From jpg1


def test_compress_images_png_to_png():
    """
    Test PNG to PNG compression (preserves alpha channel).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test PNG with alpha
        src_path = tmpdir / "image.png"
        create_test_image_png(src_path, with_alpha=True)

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format png --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Output file should be created
        dst_path = tmpdir / "processed_image.png"
        assert dst_path.exists()

        # Check file is valid PNG
        img = cv2.imread(str(dst_path), cv2.IMREAD_UNCHANGED)
        assert img is not None


def test_compress_images_preserve_extension_jpg():
    """
    Test that JPG->JPG preserves original extension (.JPG, .jpeg).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test JPG with .JPG extension (uppercase)
        src_path = tmpdir / "image.JPG"
        create_test_image_jpg(src_path)

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format jpg --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Output file should preserve .JPG extension
        dst_path = tmpdir / "processed_image.JPG"
        assert dst_path.exists()


def test_compress_images_preserve_extension_png():
    """
    Test that PNG->PNG preserves original extension.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test PNG with .PNG extension (uppercase)
        src_path = tmpdir / "image.PNG"
        create_test_image_png(src_path)

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format png --yes"
        proc = _run_command(cmd)

        # Should succeed
        assert proc.returncode == 0

        # Output file should preserve .PNG extension
        dst_path = tmpdir / "processed_image.PNG"
        assert dst_path.exists()


def test_compress_images_yes_flag():
    """
    Test that --yes flag skips confirmation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        src_path = tmpdir / "image.png"
        create_test_image_png(src_path)

        # Run command with --yes flag
        cmd = f"ct compress-images {src_path} --format jpg --yes"
        proc = _run_command(cmd)

        # Should succeed and process automatically
        assert proc.returncode == 0
        assert b"--yes specified" in proc.stdout or b"Processing" in proc.stdout

        # Output file should be created without prefix (format conversion)
        dst_path = tmpdir / "image.jpg"
        assert dst_path.exists()


def test_compress_images_cancel():
    """
    Test that user can cancel the operation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        src_path = tmpdir / "image.png"
        create_test_image_png(src_path)

        # Run command and answer 'n' to proceed
        cmd = f"ct compress-images {src_path} --format jpg"
        proc = _run_command(cmd, input_text="n\n")

        # Should succeed but not create output
        assert proc.returncode == 0
        assert b"Aborted" in proc.stdout

        # No output file should be created
        dst_path = tmpdir / "processed_image.jpg"
        assert not dst_path.exists()
