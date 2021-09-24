# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test input data."""

# =============================================================================
# IMPORTS
# =============================================================================


from galaxychop import data
from galaxychop import io


# =============================================================================
# IO TESTS
# =============================================================================


def test_read_npy(data_path):
    path_star = data_path("star_ID_394242.npy")
    path_gas = data_path("gas_ID_394242.npy")
    path_dark = data_path("dark_ID_394242.npy")

    path_pot_s = data_path("potential_star_ID_394242.npy")
    path_pot_dm = data_path("potential_dark_ID_394242.npy")
    path_pot_g = data_path("potential_gas_ID_394242.npy")

    columns = ["m", "x", "y", "z", "vx", "vy", "vz", "id"]

    gala = io.read_npy(
        path_star=path_star,
        path_dark=path_dark,
        path_gas=path_gas,
        columns=columns,
        path_pot_s=path_pot_s,
        path_pot_dm=path_pot_dm,
        path_pot_g=path_pot_g,
    )

    assert isinstance(gala, data.Galaxy) is True

    assert (
        len(gala.stars) == 32067
        and gala.stars.softening == 0.0
        and gala.stars.has_potential_
    )
    assert (
        len(gala.dark_matter) == 21156
        and gala.dark_matter.softening == 0.0
        and gala.dark_matter.has_potential_
    )

    assert (
        len(gala.gas) == 4061
        and gala.gas.softening == 0.0
        and gala.gas.has_potential_
    )


def test_read_hdf5(data_path):
    path = data_path("gal394242.h5")
    gala = io.read_hdf5(path=path)

    assert isinstance(gala, data.Galaxy) is True

    assert (
        len(gala.stars) == 32067
        and gala.stars.softening == 0.0
        and gala.stars.has_potential_
    )
    assert (
        len(gala.dark_matter) == 21156
        and gala.dark_matter.softening == 0.0
        and gala.dark_matter.has_potential_
    )

    assert (
        len(gala.gas) == 4061
        and gala.gas.softening == 0.0
        and gala.gas.has_potential_
    )
