```bash
# Default behavior (do nothing):
# No changes made to any files
ct compress-images *.png *.jpg

# --format and --inplace
# Notes:
# - When --inplace is specified, the original file will be deleted, even for
#   png -> jpg conversion.
# - When converting png -> jpg or jpg -> png, the output extension is always
#   ".jpg" or ".png".
# - When converting jpg -> jpg or png -> png, the output extension is the same as
#   the original file. E.g. ".JPG" -> ".JPG", ".jpeg" -> ".jpeg".
# png -> jpg (image.jpg created, image.png unchanged)
ct compress-images image.png --format jpg
# jpeg -> png (image.png created, image.jpeg unchanged)
ct compress-images image.jpeg --format png
# jpg -> jpg (processed_image.jpg created, image.jpg unchanged)
ct compress-images image.jpg --format jpg
# png -> jpg inplace (image.jpg created, image.png deleted)
ct compress-images image.png --format jpg --inplace
# jpg (.jpeg extension) -> jpg inplace (image.jpeg is updated)
ct compress-images image.jpeg --format jpg --inplace

# --quality
# Notes:
# - Quality only works for JPG formats, not when output format is PNG. The flag will raise an exception.
# Compress with quality 80 (processed_image.png created, image.png unchanged)
ct compress-images image.png --quality 80
# Compress with quality 90 (processed_image.jpeg created, image.jpeg unchanged)
ct compress-images image.jpeg --quality 90

# --skip_compression_ratio
# Notes:
# - Skip compression if the compression ratio is above this value (default: 1.0)
# - Only applies to JPG->JPG compression to avoid recompressing already compressed images
# - This doesn't affect the output file name, it just bypasses compression by
#   directly copying the file.
# - Default is 1.0 to process all files (no skipping)
# Skip compression if ratio > 0.95 (skip files that are hard to compress)
ct compress-images image.jpg --quality 80 --skip_compression_ratio 0.95
# Skip compression if ratio > 0.8 (compress more aggressively)
ct compress-images image.jpg --quality 80 --skip_compression_ratio 0.8

# Common use cases
# Convert recursively all .png to .jpg files, save with quality 90
ct compress-images **/*.png --format jpg --quality 90

# Compress recursively all .jpg/.jpeg/.JPG/.JPEG to .jpg to quality 90, inplace,
# skip if the compression ratio is above 0.90. The original file name and
# extension are preserved.
ct compress-images **/*.jpg **/*.jpeg **/*.JPG **/*.JPEG --quality 90 --skip_compression_ratio 0.9 --inplace

```

## Removed flags (to implement)

`--update_texts_in_dir`
This flag is completely removed as we don't support this feature anymore.

`--png_only`
This flag is completely removed as we don't support this feature anymore.

`--flatten_alpha_channel`
This flag is completely removed as we don't support this feature anymore.
Instead, we define new default behaviors:
- PNG -> JPG: Alpha channel is removed (flattened to white background)
- PNG -> PNG: Alpha channel is preserved
