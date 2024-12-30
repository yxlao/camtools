# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import subprocess
import camtools as ct
from pathlib import Path
import shutil

sys.path.insert(0, os.path.abspath(".."))

# Get version from camtools package
version = ct.__version__

# Get git commit hash
try:
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    release = f"{version}+{git_hash}"
except subprocess.CalledProcessError:
    release = version

project = "CamTools"
copyright = "2024, Yixing Lao"
author = "Yixing Lao"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static", "../camtools/assets"]

# Get script directory
script_dir = Path(__file__).parent

# Theme options
html_theme_options = {
    "light_logo": "camtools_logo_light.png",
    "dark_logo": "camtools_logo_dark.png",
}

# Favicon
favicon_path = (
    script_dir.parent / "camtools" / "assets" / "camtools_logo_squre_dark.png"
)
if not favicon_path.is_file():
    raise FileNotFoundError(f"Favicon not found at {favicon_path}")
html_favicon = str(favicon_path)


# Google Analytics configuration
def add_ga_javascript(app, pagename, templatename, context, doctree):
    """
    Ref: https://github.com/sphinx-contrib/googleanalytics/blob/master/sphinxcontrib/googleanalytics.py
    """
    metatags = context.get("metatags", "")
    metatags += """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-FG59NQBWRW"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-FG59NQBWRW');
    </script>
    """
    context["metatags"] = metatags


def setup(app):
    app.connect("html-page-context", add_ga_javascript)


# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# AutoDoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
add_module_names = True
python_use_unqualified_type_names = False

# Custom module name display
modindex_common_prefix = ["camtools."]  # Strip 'camtools.' from module index

sys.modules["ct"] = ct  # Allow using 'ct' as an alias in documentation

# Suppress specific warnings
suppress_warnings = [
    "toctree.excluded",  # Suppress warnings about files not in any toctree
]
