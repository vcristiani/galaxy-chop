# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute galaxychop."""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import setuptools  # noqa

from skbuild import setup

# =============================================================================
# PATH TO THIS MODULE
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

# =============================================================================
# Get the version from galaxy-chop file itself (not imported)
# =============================================================================

galaxychop_INIT_PATH = os.path.join(PATH, "galaxychop", "__init__.py")

with open(galaxychop_INIT_PATH, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy >= 1.13.3",
    "scipy >= 1.0",
    "scikit-learn",
    "astropy",
    "uttrs",
    "pandas",
    "h5py",
    "custom_inherit",
    "seaborn",
]

# =============================================================================
# DESCRIPTION
# =============================================================================

with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()

# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="galaxychop",
    version=VERSION,
    description="Galaxy dynamic de-composition",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Valeria Cristiani et al",
    author_email="valeria.cristiani@unc.edu.ar",
    url="https://github.com/vcristiani/galaxy-chop",
    packages=[
        "galaxychop",
        "galaxychop.models",
        "galaxychop.preproc",
        "galaxychop.preproc.fortran",
        "galaxychop.utils",
    ],
    license="MIT",
    keywords="galaxy, dynamics",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=REQUIREMENTS,
)
