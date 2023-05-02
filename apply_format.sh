#!/usr/bin/env bash

# Check if yapf is installed.
if ! command -v yapf >/dev/null; then
    echo "yapf not in path, can not format. Please install yapf:"
    echo "    pip install yapf"
    exit 2
fi

# Apply style.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find "${script_dir}" -name '*.py' -print0 | xargs -0 yapf -i
echo "Style applied for ${script_dir}."
