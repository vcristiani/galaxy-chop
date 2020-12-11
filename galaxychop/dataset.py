# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Load tutorial files Module."""

# #####################################################
# IMPORTS
# #####################################################

import os
from pathlib import Path

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PATH = Path(os.path.abspath(os.path.dirname(__file__)))

TEST_DATA_PATH = PATH / "tutorialdata"


# #####################################################
# FUNCTIONS
# #####################################################


def load_star():
    """Input for testing."""
    path = TEST_DATA_PATH / "star.dat"

    return np.loadtxt(path)


def load_dark():
    """Input for testing."""
    path = TEST_DATA_PATH / "dark.dat"

    return np.loadtxt(path)


def load_gas():
    """Input for testing."""
    path = TEST_DATA_PATH / "gas_.dat"

    return np.loadtxt(path)
