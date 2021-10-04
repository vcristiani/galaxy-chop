# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities provided by galaxychop."""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import data, utils

import numpy as np

import pytest


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

    assert isinstance(pgal, data.Galaxy)
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
        assert not (ocol == ccol).all()


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


# =============================================================================
# ALIGN
# =============================================================================


def test_star_align_rcur0dot9(galaxy):
    gal = galaxy(seed=42)

    agal = utils.star_align(gal, r_cut=0.9)

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

    agal = utils.star_align(gal)

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
        utils.star_align(gal, r_cut=-1)


def test_is_star_aligned_real_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    agal = utils.star_align(gal, r_cut=5)

    assert not utils.is_star_aligned(gal, r_cut=5)
    assert utils.is_star_aligned(agal, r_cut=5)


def test_is_star_aligned_fake_galaxy(galaxy):
    gal = galaxy(seed=42)

    agal = utils.star_align(gal, r_cut=5)

    assert not utils.is_star_aligned(gal, r_cut=5)
    assert utils.is_star_aligned(agal, r_cut=5)


# =============================================================================
# JCIRC
# =============================================================================


def test_jcirc_real_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = utils.jcirc(gal)

    mask_energy = np.where(~np.isnan(result.E_star_norm))[0]
    mask_eps = np.where(~np.isnan(result.eps))[0]

    assert (result.E_star_norm[mask_energy] != np.nan).all()
    assert (result.E_star_norm[mask_energy] <= 0).all()
    assert (result.eps[mask_eps] != np.nan).all()
    assert (result.eps[mask_eps] <= 1).all()
    assert (result.eps[mask_eps] >= -1).all()


@pytest.mark.xfail
def test_jcirc_fake_galaxy(galaxy):
    gal = galaxy(seed=42)
    result = utils.jcirc(gal)

    mask_energy = np.where(~np.isnan(result.E_star_norm))[0]
    mask_eps = np.where(~np.isnan(result.eps))[0]

    assert (result.E_star_norm[mask_energy] != np.nan).all()
    assert (result.E_star_norm[mask_energy] <= 0).all()
    assert (result.eps[mask_eps] != np.nan).all()
    assert (result.eps[mask_eps] <= 1).all()
    assert (result.eps[mask_eps] >= -1).all()


def test_x_y_len(read_hdf5_galaxy):
    """Check the x and y array len."""
    gal = read_hdf5_galaxy("gal394242.h5")
    result = utils.jcirc(gal)

    x = result.x
    y = result.y

    assert len(x) == len(y)

