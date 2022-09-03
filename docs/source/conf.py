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
    "sphinxfortran.fortran_domain",
    "sphinxfortran.fortran_autodoc",
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

import m2r2

DYNAMIC_RST = {
    "README.md": "README.rst",
    "CHANGELOG.md": "CHANGELOG.rst",
}

for md_name, rst_name in DYNAMIC_RST.items():
    md_path = CHOP_PATH / md_name
    with open(md_path) as fp:
        readme_md = fp.read().split("<!-- BODY -->")[-1]

    rst_path = CURRENT_PATH / "_dynamic" / rst_name

    with open(rst_path, "w") as fp:
        fp.write(".. FILE AUTO GENERATED !! \n")
        fp.write(m2r2.convert(readme_md))
        print(f"{md_path} -> {rst_path} regenerated!")


# =============================================================================
# FORTRAN
# =============================================================================

## -- Options for Sphinx-Fortran ---------------------------------------------
# List of possible extensions in the case of a directory listing
fortran_ext = ['f90', 'F90', 'f95', 'F95']

# This variable must be set with file pattern, like "*.f90", or a list of them.
# It is also possible to specify a directory name; in this case, all files than
# have an extension matching those define by the config variable `fortran_ext`
# are used.
fortran_src = [ str(CHOP_PATH / "galaxychop" / "preproc" / "fortran"),  ]

# Indentation string or length (default 4). If it is an integer,
# indicates the number of spaces.
fortran_indent = 4

# =============================================================================
# SETUP
# =============================================================================


def setup(app):
    app.add_css_file("css/galaxychop.css")
    app.add_js_file("js/galaxychop.js")
