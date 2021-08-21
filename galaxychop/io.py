# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module galaxy-chop."""

# =============================================================================
# IMPORTS
# =============================================================================


from astropy.table import Table

import h5py

import numpy as np

import pandas as pd

from . import core

# =============================================================================
# UTILS
# =============================================================================


def _table_to_dict(table, key_suffix):
    kws = {f"{k}_{key_suffix}": v for k, v in table.items() if k != "id"}
    kws[f"pot_{key_suffix}"] = kws.pop(f"potential_{key_suffix}", None)

    return kws

# =============================================================================
# API
# =============================================================================

def read_hdf5(
    path,
    softening_s: float = 0.0,
    softening_dm: float = 0.0,
    softening_g: float = 0.0,
):

    with h5py.File(path, "r") as f:
        star_table = Table.read(f["stars"])
        dark_table = Table.read(f["dark_matter"])
        gas_table = Table.read(f["gas"])

    galaxy_kws = {
        "softening_s": softening_s,
        "softening_dm": softening_dm,
        "softening_g": softening_g,
    }

    star_kws = _table_to_dict(star_table, "s")
    galaxy_kws.update(star_kws)

    dark_kws = _table_to_dict(dark_table, "dm")
    galaxy_kws.update(dark_kws)

    gas_kws = _table_to_dict(gas_table, "g")
    galaxy_kws.update(gas_kws)

    galaxy = core.mkgalaxy(**galaxy_kws)

    return galaxy


def read_file(
    path_star,
    path_dark,
    path_gas,
    columns,
    path_pot_s=None,
    path_pot_dm=None,
    path_pot_g=None,
    softening_s: float = 0.0,
    softening_dm: float = 0.0,
    softening_g: float = 0.0,
):

    particles_star = np.load(path_star)
    particles_dark = np.load(path_dark)
    particles_gas = np.load(path_gas)

    df_star = pd.DataFrame(particles_star, columns=columns)
    df_dark = pd.DataFrame(particles_dark, columns=columns)
    df_gas = pd.DataFrame(particles_gas, columns=columns)

    if path_pot_s is not None:
        pot_s = np.load(path_pot_s)
        df_star.insert(7, "potential", pot_s, True)

    if path_pot_dm is not None:
        pot_dm = np.load(path_pot_dm)
        df_dark.insert(7, "potential", pot_dm, True)

    if path_pot_g is not None:
        pot_g = np.load(path_pot_g)
        df_gas.insert(7, "potential", pot_g, True)

    star_table = Table.from_pandas(df_star)
    dark_table = Table.from_pandas(df_dark)
    gas_table = Table.from_pandas(df_gas)

    galaxy_kws = {
        "softening_s": softening_s,
        "softening_dm": softening_dm,
        "softening_g": softening_g,
    }

    star_kws = _table_to_dict(star_table, "s")
    galaxy_kws.update(star_kws)

    dark_kws = _table_to_dict(dark_table, "dm")
    galaxy_kws.update(dark_kws)

    gas_kws = _table_to_dict(gas_table, "g")
    galaxy_kws.update(gas_kws)

    galaxy = core.mkgalaxy(**galaxy_kws)

    return galaxy
