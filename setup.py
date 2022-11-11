import ctypes
from distutils.extension import Extension
from pathlib import Path
from setuptools import setup
import re

_pwd = Path(__file__).parent.absolute()


def main():
    cmdclass = dict()

    version = None
    with open(_pwd / "camtools" / "version.py") as f:
        lines = f.readlines()
        for line in lines:
            match_res = re.match(r'^__version__ = "(.*)"', line)
            if match_res:
                version = match_res.group(1)
                break
    if version is None:
        raise RuntimeError("Cannot find version from camtools/version.py")
    print(f"Detected version: {version}")

    status = setup(
        name="camtools",
        version=version,
        description="Camtools: Camera Tools for Computer Vision.",
        packages=['camtools'],
        cmdclass=cmdclass,
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
