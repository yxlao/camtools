from pathlib import Path
from setuptools import setup


def main():
    cmdclass = dict()

    entry_points = {
        "console_scripts": [
            "ct = camtools.tools.cli:main",
        ]
    }

    status = setup(
        name="camtools",
        description="Camtools: Camera Tools for Computer Vision.",
        packages=["camtools"],
        cmdclass=cmdclass,
        entry_points=entry_points,
        include_package_data=True,
    )


if __name__ == "__main__":
    main()
