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


# #####################################################
# FUNCTIONS
# #####################################################


def load_star():
    """Input for testing."""
    path = PATH / "star.dat"

    return np.loadtxt(path)


def load_dark():
    """Input for testing."""
    path = PATH / "dark.dat"

    return np.loadtxt(path)


def load_gas():
    """Input for testing."""
    path = PATH / "gas_.dat"

    return np.loadtxt(path)


def load_star_394242():
    """Input for testing."""
    path = PATH / "star_ID_394242.npy"

    return np.load(path)


def load_dark_394242():
    """Input for testing."""
    path = PATH / "dark_ID_394242.npy"

    return np.load(path)


def load_gas_394242():
    """Input for testing."""
    path = PATH / "gas_ID_394242.npy"

    return np.load(path)


def load_pot_star_394242():
    """Input for testing."""
    path = PATH / "potential_star_ID_394242.npy"

    return np.load(path)


def load_pot_dark_394242():
    """Input for testing."""
    path = PATH / "potential_dark_ID_394242.npy"

    return np.load(path)


def load_pot_gas_394242():
    """Input for testing."""
    path = PATH / "potential_gas_ID_394242.npy"

    return np.load(path)
