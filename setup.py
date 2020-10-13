#!/usr/bin/env python
# -*- coding: utf-8 -*-


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute galaxy-chop
"""


# =============================================================================
# IMPORTS
# =============================================================================

import sys
import os
import setuptools

from ez_setup import use_setuptools

use_setuptools()


# =============================================================================
# PATH TO THIS MODULE
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

# =============================================================================
# Get the version from galaxy-chop file itself (not imported)
# =============================================================================

galaxy-chop_INIT_PATH = os.path.join(PATH, "galaxy-chop", "__init__.py")

with open(galaxy-chop_INIT_PATH, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, PI_VERSION = line.replace('"', "").split()
            break

# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy >= 1.13.2",
    "scipy >= 1.0"
]

# =============================================================================
# DESCRIPTION
# =============================================================================
with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()

# =============================================================================
# FUNCTIONS
# =============================================================================
print(setuptools.find_packages())  # exclude=['test*']


def do_setup():
    setuptools.setup(
        name="galaxy-chop",
        version=PI_VERSION,
        description="Galaxy dynamic de-composition",
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author="Valeria Cristiani et al",
        author_email="valeria.cristiani@unc.edu.ar",
        url="https://github.com/vcristiani/galaxy-chop",
        py_modules=["ez_setup"],
        license="MIT",
        keywords="galaxy, dynamics",
        classifiers=(
            "Development Status :: 1 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
        ),
        packages=setuptools.find_packages(),  # exclude=['test*']),
        install_requires=REQUIREMENTS,
    )


def do_publish():
    pass


if __name__ == "__main__":
    if sys.argv[-1] == "publish":
        do_publish()
    else:
        do_setup()
