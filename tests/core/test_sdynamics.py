# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Test utilities  galaxychop.preproc.circ"""

# =============================================================================
# IMPORTS
# =============================================================================

import astropy.units as u

from galaxychop.core import NoGravitationalPotentialError, mkgalaxy, sdynamics

import numpy as np

import pytest


# =============================================================================
# JCIRC
# =============================================================================


def test_stellar_dynamics_real_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = sdynamics.stellar_dynamics(gal)

    mask_energy = np.where(~np.isnan(result.normalized_star_energy))[0]
    mask_eps = np.where(~np.isnan(result.eps))[0]

    assert np.all(result.normalized_star_energy[mask_energy] <= 0)
    assert np.all(result.eps[mask_eps] != np.nan)
    assert np.all(result.eps[mask_eps] <= 1)
    assert np.all(result.eps[mask_eps] >= -1)


def test_stellar_dynamics_without_potential(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal_dict = gal.disassemble()
    gal_dict.update(potential_s=None, potential_dm=None, potential_g=None)
    gal = mkgalaxy(**gal_dict)

    with pytest.raises(NoGravitationalPotentialError):
        sdynamics.stellar_dynamics(gal)


def test_stellar_dynamics_real_galaxy_reasign(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = sdynamics.stellar_dynamics(gal, reassign=True)

    mask_energy = np.where(~np.isnan(result.normalized_star_energy))[0]
    mask_eps = np.where(~np.isnan(result.eps))[0]

    assert np.all(result.normalized_star_energy[mask_energy] <= 0)
    assert np.all(result.eps[mask_eps] != np.nan)
    assert np.all(result.eps[mask_eps] <= 1)
    assert np.all(result.eps[mask_eps] >= -1)


def test_stellar_dynamics_real_galaxy_with_infinite_energy(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")

    gal.stars.total_energy_.setflags(write=True)
    gal.dark_matter.total_energy_.setflags(write=True)
    gal.gas.total_energy_.setflags(write=True)

    gal.stars.total_energy_[0] = -np.inf * u.km**2 / u.s**2
    gal.dark_matter.total_energy_[0] = -np.inf * u.km**2 / u.s**2
    gal.gas.total_energy_[0] = -np.inf * u.km**2 / u.s**2

    gal.stars.total_energy_.setflags(write=False)
    gal.dark_matter.total_energy_.setflags(write=False)
    gal.gas.total_energy_.setflags(write=False)

    result = sdynamics.stellar_dynamics(gal)

    mask_energy = np.where(~np.isnan(result.normalized_star_energy))[0]
    mask_eps = np.where(~np.isnan(result.eps))[0]

    assert np.all(result.normalized_star_energy[mask_energy] <= 0)
    assert np.all(result.eps[mask_eps] != np.nan)
    assert np.all(result.eps[mask_eps] <= 1)
    assert np.all(result.eps[mask_eps] >= -1)


def test_stellar_dynamics_fake_galaxy(galaxy):
    for seed in range(100):
        gal = galaxy(seed=seed)
        with pytest.raises(ValueError):
            sdynamics.stellar_dynamics(gal)


def test_GalaxyStellarDynamics_x_y_len(read_hdf5_galaxy):
    """Check the x and y array len."""
    gal = read_hdf5_galaxy("gal394242.h5")
    result = sdynamics.stellar_dynamics(gal)

    x = result.x
    y = result.y

    assert len(x) == len(y)


def test_GalaxyStellarDynamics_x(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = sdynamics.stellar_dynamics(gal)

    aux0 = np.zeros(len(result.x) + 1)
    aux1 = np.zeros(len(result.x) + 1)

    aux0[1:] = result.x
    aux1[: len(result.x)] = result.x

    diff = aux1[1:] - aux0[1:]

    assert (diff > 0).all()


def test_GalaxyStellarDynamics_y(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = sdynamics.stellar_dynamics(gal)

    df = gal.to_dataframe(attributes=["total_energy", "Jz"])

    E = df.total_energy.values
    Jz = df.Jz.values
    mask_bound = np.where((E <= 0.0) & (E != -np.inf))[0]
    Jz_max = np.max(np.abs(Jz[mask_bound]))

    y_result = np.abs(Jz[mask_bound]) / Jz_max

    assert np.isin(result.y, y_result).all()


def test_GalaxyStellarDynamics_repr(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    result = repr(sdynamics.stellar_dynamics(gal))
    expected = (
        "<_GalaxyStellarDynamics normalized_star_energy=37393, "
        "normalized_star_Jz=37393, eps=37393, eps_r=37393, x=39, y=39>"
    )
    assert result == expected
