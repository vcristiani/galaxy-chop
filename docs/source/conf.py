# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import pathlib


# this path is pointing to project/docs/source
CURRENT_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
CHOP_PATH = CURRENT_PATH.parent.parent

sys.path.insert(0, str(CHOP_PATH))


import galaxychop


# -- Project information -----------------------------------------------------

project = "galaxy-chop"
copyright = "2020, 2021, Valeria Cristiani"
author = "Valeria Cristiani"

# The full version, including alpha/beta/rc tags
release = galaxychop.__version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"

# -- Options for nbsphinx output ----------------------------------------------
nbsphinx_prompt_width = "0pt"


# =============================================================================
# INJECT README INTO THE RESTRUCTURED TEXT index.rst
# =============================================================================

import m2r

with open(CHOP_PATH / "README.md") as fp:
    readme_md = fp.read().split("<!-- BODY -->")[-1]


README_RST_PATH = CURRENT_PATH / "_dynamic" / "README"


with open(README_RST_PATH, "w") as fp:
    fp.write(".. FILE AUTO GENERATED !! \n")
    fp.write(m2r.convert(readme_md))
    print(f"{README_RST_PATH} regenerated!")
