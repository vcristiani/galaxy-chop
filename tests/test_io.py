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
PATH = Path(os.path.abspath(os.path.dirname(".")))

# =============================================================================
# IO TESTS
# =============================================================================
path_star = str(PATH) + "/galaxychop/dataset/star_ID_394242.npy"
path_gas = str(PATH) + "/galaxychop/dataset/gas_ID_394242.npy"
path_dark = str(PATH) + "/galaxychop/dataset/dark_ID_394242.npy"

columns=['m', 'x', 'y', 'z', 'vx', 'vy', 'vz','id']

path_pot_s = str(PATH) + "/galaxychop/dataset/potential_star_ID_394242.npy"
path_pot_dm = str(PATH) + "/galaxychop/dataset/potential_dark_ID_394242.npy"
path_pot_g = str(PATH) + "/galaxychop/dataset/potential_gas_ID_394242.npy"

def test_read_file_method():
    gala = io.read_file(
        path_star=path_star,
        path_dark=path_dark,
        path_gas=path_gas,
        columns=columns,
        path_pot_s=path_pot_s,
        path_pot_dm=path_pot_dm,
        path_pot_g=path_pot_g
    )

    assert isinstance(gala, core.Galaxy) == True
    
def test_read_hdf5_method():
    path = str(PATH) + "/galaxychop/dataset/gal394242.h5"
    gala = io.read_hdf5(path=path)

    assert isinstance(gala, core.Galaxy) == True