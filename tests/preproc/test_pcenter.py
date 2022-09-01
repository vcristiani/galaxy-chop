# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities provided by galaxychop."""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop.preproc import pcenter



import pytest


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
        pcenter.center(gal)


def test_center(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )

    cgal = pcenter.center(gal)

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
        assert not (ocol == ccol).all()


def test_is_centered_without_potential_energy(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )
    with pytest.raises(ValueError):
        pcenter.is_centered(gal)


def test_is_centered(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )

    cgal = pcenter.center(gal)

    assert not pcenter.is_centered(gal)
    assert pcenter.is_centered(cgal)
