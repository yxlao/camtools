#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Check that the setup.py file exists.
if [ ! -f "$script_dir/setup.py" ]; then
    echo "setup.py does not exist."
    echo "Please run this script in the root directory of the project."
    exit 1
fi

# Check that the camtools dir exists.
if [ ! -d "$script_dir/camtools" ]; then
    echo "camtools dir does not exist."
    echo "Please run this script in the root directory of the project."
    exit 1
fi

# Check that conda is installed.
if ! [ -x "$(command -v conda)" ]; then
    echo "conda is not installed."
    echo "Please install conda first."
    exit 1
fi

# Check that conda is not activated.
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "$CONDA_DEFAULT_ENV is activated."
    echo "Please deactivate conda environment first."
    exit 1
fi

# Check that ~/.pypirc exists.
if [ ! -f "$HOME/.pypirc" ]; then
    echo "$HOME/.pypirc does not exist."
    echo "Please configure the .pypirc file first:"
    echo "######"
    echo "[distutils]"
    echo "index-servers ="
    echo "    pypi"
    echo "    testpypi"
    echo ""
    echo "[pypi]"
    echo "repository = https://upload.pypi.org/legacy/"
    echo "username = __token__"
    echo "password = pypi-xxxx"
    echo ""
    echo "[testpypi]"
    echo "repository = https://test.pypi.org/legacy/"
    echo "username = __token__"
    echo "password = pypi-xxxx"
    echo "######"
    exit 1
fi

clear_up_dirs=(
    .pytest_cache
    build
    camtools.egg-info
    dist
)
for dir in "${clear_up_dirs[@]}"; do
    if [ -d "$script_dir/$dir" ]; then
        echo "Removing: $script_dir/$dir"
        rm -rf "$script_dir/$dir"
    fi
done

# Create conda env.
rm -rf /tmp/ct
conda create --prefix /tmp/ct python=3.9 -y

# Activate conda env.
eval "$(conda shell.bash hook)"
conda activate /tmp/ct

# Check that conda is activated.
if [ "$CONDA_DEFAULT_ENV" != "/tmp/ct" ]; then
    echo "Conda environment ct is not activated. Erorr."
    exit 1
else
    echo "Conda environment ct is activated."
fi

# Check that python is from the conda env.
if [ "$(which python)" != "/tmp/ct/bin/python" ]; then
    echo "python is not /tmp/ct/bin/python."
    echo "Please activate conda environment ct first."
    exit 1
else
    echo "Python: $(which python)"
fi

# Install requirements.
python -m pip install build twine

# Build.
python -m build

# List files in dist.
echo "Going to upload the following files:"
for file in dist/*; do
    echo "> $file: $(du -h $file | cut -f1)"
done

# Upload.
read -r -p "Run twine upload? [y/N] " response
case "$response" in
[yY][eE][sS] | [yY])
    echo "twine upload -r pypi dist/*"
    twine upload -r pypi dist/*
    ;;
*)
    echo "Aborted."
    ;;
esac
