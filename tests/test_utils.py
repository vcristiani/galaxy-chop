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
import numpy.testing as npt

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


def test_Galaxy_potential_energy_fortran_backend(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal_f = utils.potential(gal, backend="fortran")

    assert isinstance(pgal_f, data.Galaxy)
    assert np.all(pgal_f.stars.potential == pgal_f.potential_energy_[0])
    assert np.all(pgal_f.dark_matter.potential == pgal_f.potential_energy_[1])
    assert np.all(pgal_f.gas.potential == pgal_f.potential_energy_[2])


def test_Galaxy_potential_energy_backend_consistency(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal_np = utils.potential(gal, backend="numpy")
    pgal_f = utils.potential(gal, backend="fortran")

    decimal = 5
    npt.assert_almost_equal(
        pgal_np.stars.potential.value, pgal_f.stars.potential.value, decimal
    )
    npt.assert_almost_equal(
        pgal_np.dark_matter.potential.value,
        pgal_f.dark_matter.potential.value,
        decimal,
    )
    npt.assert_almost_equal(
        pgal_np.gas.potential.value, pgal_f.gas.potential.value, decimal
    )


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
            utils.jcirc(gal)


def test_x_y_len(read_hdf5_galaxy):
    """Check the x and y array len."""
    gal = read_hdf5_galaxy("gal394242.h5")
    result = utils.jcirc(gal)

    x = result.x
    y = result.y

    assert len(x) == len(y)


def test_x_values(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = utils.jcirc(gal)

    aux0 = np.zeros(len(result.x) + 1)
    aux1 = np.zeros(len(result.x) + 1)

    aux0[1:] = result.x
    aux1[: len(result.x)] = result.x

    diff = aux1[1:] - aux0[1:]

    assert (diff > 0).all()


def test_y_values(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = utils.jcirc(gal)

    df = gal.to_dataframe(attributes=["total_energy", "Jz"])

    E = df.total_energy.values
    Jz = df.Jz.values
    mask_bound = np.where((E <= 0.0) & (E != -np.inf))[0]
    Jz_max = np.max(np.abs(Jz[mask_bound]))

    y_result = np.abs(Jz[mask_bound]) / Jz_max

    assert np.isin(result.y, y_result).all()
