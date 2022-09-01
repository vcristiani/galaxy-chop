# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities  galaxychop.preproc.circ"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop.preproc import circ

import numpy as np

import pytest

# =============================================================================
# JCIRC
# =============================================================================


def test_jcirc_real_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = circ.jcirc(gal)

    mask_energy = np.where(~np.isnan(result.normalized_star_energy))[0]
    mask_eps = np.where(~np.isnan(result.eps))[0]

    assert np.all(result.normalized_star_energy[mask_energy] <= 0)
    assert np.all(result.eps[mask_eps] != np.nan)
    assert np.all(result.eps[mask_eps] <= 1)
    assert np.all(result.eps[mask_eps] >= -1)


def test_jcirc_fake_galaxy(galaxy):
    for seed in range(100):
        gal = galaxy(seed=seed)
        with pytest.raises(ValueError):
            circ.jcirc(gal)


def test_x_y_len(read_hdf5_galaxy):
    """Check the x and y array len."""
    gal = read_hdf5_galaxy("gal394242.h5")
    result = circ.jcirc(gal)

    x = result.x
    y = result.y

    assert len(x) == len(y)


def test_x_values(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = circ.jcirc(gal)

    aux0 = np.zeros(len(result.x) + 1)
    aux1 = np.zeros(len(result.x) + 1)

    aux0[1:] = result.x
    aux1[: len(result.x)] = result.x

    diff = aux1[1:] - aux0[1:]

    assert (diff > 0).all()


def test_y_values(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = circ.jcirc(gal)

    df = gal.to_dataframe(attributes=["total_energy", "Jz"])

    E = df.total_energy.values
    Jz = df.Jz.values
    mask_bound = np.where((E <= 0.0) & (E != -np.inf))[0]
    Jz_max = np.max(np.abs(Jz[mask_bound]))

    y_result = np.abs(Jz[mask_bound]) / Jz_max

    assert np.isin(result.y, y_result).all()
