import ctypes
from distutils.extension import Extension
from glob import glob
from pathlib import Path
from setuptools import setup
import re


def main():
    # Force platform specific wheel.
    # https://stackoverflow.com/a/45150383/1255535
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

        class bdist_wheel(_bdist_wheel):

            def finalize_options(self):
                _bdist_wheel.finalize_options(self)
                self.root_is_pure = False

            def get_tag(self):
                python, abi, plat = _bdist_wheel.get_tag(self)
                if plat == 'linux_x86_64':
                    libc = ctypes.CDLL('libc.so.6')
                    libc.gnu_get_libc_version.restype = ctypes.c_char_p
                    GLIBC_VER = libc.gnu_get_libc_version().decode(
                        'utf8').split('.')
                    plat = f'manylinux_{GLIBC_VER[0]}_{GLIBC_VER[1]}_x86_64'
                return python, abi, plat

    except ImportError:
        print('Warning: cannot import "wheel" to build platform-specific wheel')
        bdist_wheel = None

    if bdist_wheel is None:
        cmdclass = dict()
    else:
        cmdclass = {"bdist_wheel": bdist_wheel}

    version = None
    with open("camtools/version.py") as f:
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
