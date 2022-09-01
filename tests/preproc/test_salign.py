# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities  galaxychop.preproc.salign"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop.preproc import salign



import pytest


# =============================================================================
# ALIGN
# =============================================================================


def test_star_align_rcur0dot9(galaxy):
    gal = galaxy(seed=42)

    agal = salign.star_align(gal, r_cut=0.9)

    df = gal.to_dataframe()
    adf = agal.to_dataframe()

    changed = [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "Jx",
        "Jy",
        "Jz",
        "kinetic_energy",
        "total_energy",
    ]

    for colname in df.columns[~df.columns.isin(changed)]:
        ocol = df[colname]
        acol = adf[colname]
        assert (ocol == acol).all(), colname

    for colname in changed:
        ocol = df[colname]
        acol = adf[colname]
        assert not (ocol == acol).all(), colname


def test_star_align(galaxy):
    gal = galaxy(seed=42)

    agal = salign.star_align(gal)

    df = gal.to_dataframe()
    adf = agal.to_dataframe()

    changed = [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "Jx",
        "Jy",
        "Jz",
        "kinetic_energy",
        "total_energy",
    ]

    for colname in df.columns[~df.columns.isin(changed)]:
        ocol = df[colname]
        acol = adf[colname]
        assert (ocol == acol).all(), colname

    for colname in changed:
        ocol = df[colname]
        acol = adf[colname]
        assert not (ocol == acol).all(), colname


def test_star_align_invalid_rcut(galaxy):
    gal = galaxy(seed=42)

    with pytest.raises(ValueError):
        salign.star_align(gal, r_cut=-1)


def test_is_star_aligned_real_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    agal = salign.star_align(gal, r_cut=5)

    assert not salign.is_star_aligned(gal, r_cut=5)
    assert salign.is_star_aligned(agal, r_cut=5)


def test_is_star_aligned_fake_galaxy(galaxy):
    gal = galaxy(seed=42)

    agal = salign.star_align(gal, r_cut=5)

    assert not salign.is_star_aligned(gal, r_cut=5)
    assert salign.is_star_aligned(agal, r_cut=5)
