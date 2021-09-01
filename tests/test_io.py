# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test input data."""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import core
from galaxychop import io

import os
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================
PATH = Path(os.path.abspath(os.path.dirname(__file__)))
TEST_DATA_PATH = PATH / "test_data"


# =============================================================================
# IO TESTS
# =============================================================================


def test_read_file_method():
    path_star = TEST_DATA_PATH / "star_ID_394242.npy"
    path_gas = TEST_DATA_PATH / "gas_ID_394242.npy"
    path_dark = TEST_DATA_PATH / "dark_ID_394242.npy"

    columns = ["m", "x", "y", "z", "vx", "vy", "vz", "id"]

    path_pot_s = TEST_DATA_PATH / "potential_star_ID_394242.npy"
    path_pot_dm = TEST_DATA_PATH / "potential_dark_ID_394242.npy"
    path_pot_g = TEST_DATA_PATH / "potential_gas_ID_394242.npy"

    gala = io.read_file(
        path_star=path_star,
        path_dark=path_dark,
        path_gas=path_gas,
        columns=columns,
        path_pot_s=path_pot_s,
        path_pot_dm=path_pot_dm,
        path_pot_g=path_pot_g,
    )

    assert isinstance(gala, core.Galaxy) is True


def test_read_hdf5_method():
    path = TEST_DATA_PATH / "gal394242.h5"
    gala = io.read_hdf5(path=path)

    assert isinstance(gala, core.Galaxy) is True
