# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test input data."""

# =============================================================================
# IMPORTS
# =============================================================================

from io import BytesIO

from galaxychop import data
from galaxychop import io

import pandas as pd


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
        path_or_stream_star=path_star,
        path_or_stream_dark=path_dark,
        path_or_stream_gas=path_gas,
        columns=columns,
        path_or_stream_pot_s=path_pot_s,
        path_or_stream_pot_dm=path_pot_dm,
        path_or_stream_pot_g=path_pot_g,
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


def test_read_npy_stream(data_path):
    path_star = data_path("star_ID_394242.npy")
    path_gas = data_path("gas_ID_394242.npy")
    path_dark = data_path("dark_ID_394242.npy")

    path_pot_s = data_path("potential_star_ID_394242.npy")
    path_pot_dm = data_path("potential_dark_ID_394242.npy")
    path_pot_g = data_path("potential_gas_ID_394242.npy")

    columns = ["m", "x", "y", "z", "vx", "vy", "vz", "id"]

    gala = io.read_npy(
        path_or_stream_star=open(path_star, "rb"),
        path_or_stream_dark=open(path_dark, "rb"),
        path_or_stream_gas=open(path_gas, "rb"),
        columns=columns,
        path_or_stream_pot_s=open(path_pot_s, "rb"),
        path_or_stream_pot_dm=open(path_pot_dm, "rb"),
        path_or_stream_pot_g=open(path_pot_g, "rb"),
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
    gala = io.read_hdf5(path)

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


def test_read_hdf5_stream(data_path):
    path = data_path("gal394242.h5")
    with open(path, "rb") as fp:
        gala = io.read_hdf5(fp)

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


def test_to_hdf5(galaxy):

    gal = galaxy(seed=42)

    buff = BytesIO()
    io.to_hdf5(buff, gal)
    buff.seek(0)
    result = io.read_hdf5(buff)

    stored_attributes = ["m", "x", "y", "z", "vx", "vy", "vz"]

    result_df = result.to_dataframe(attributes=stored_attributes)
    expected_df = gal.to_dataframe(attributes=stored_attributes)

    pd.testing.assert_frame_equal(result_df, expected_df)
