# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities provided by galaxychop."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from galaxychop import core, utils

# =============================================================================
#   POTENTIAL ENERGY
# =============================================================================


def test_Galaxy_potential_energy_already_calculated(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )
    with pytest.raises(ValueError):
        utils.potential(gal)


def test_Galaxy_potential_energy(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal = utils.potential(gal)

    assert isinstance(pgal, core.Galaxy)
    assert np.all(pgal.stars.potential == pgal.potential_energy_[0])
    assert np.all(pgal.dark_matter.potential == pgal.potential_energy_[1])
    assert np.all(pgal.gas.potential == pgal.potential_energy_[2])


# =============================================================================
# CENTER
# =============================================================================


def test_center_without_potential_energy(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )
    with pytest.raises(ValueError):
        utils.center(gal)


def test_center(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )

    cgal = utils.center(gal)

    df = gal.to_dataframe()
    cdf = cgal.to_dataframe()

    changed = ["x", "y", "z", "Jx", "Jy", "Jz"]

    for colname in df.columns[~df.columns.isin(changed)]:
        ocol = df[colname]
        ccol = cdf[colname]
        assert (ocol == ccol).all()

    for colname in changed:
        ocol = df[colname]
        ccol = cdf[colname]
        assert (ocol == ccol).all() == False


def test_is_centered_without_potential_energy(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )
    with pytest.raises(ValueError):
        utils.is_centered(gal)


def test_is_centered(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )

    cgal = utils.center(gal)

    assert not utils.is_centered(gal)
    assert utils.is_centered(cgal)
