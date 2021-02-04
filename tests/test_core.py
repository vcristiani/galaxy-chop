# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test input data."""

# =============================================================================
# IMPORTS
# =============================================================================

import astropy.units as u

import dask.array as da

from galaxychop import core
from galaxychop import utils

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    "shorten",
    [
        "m_s",
        "x_s",
        "y_s",
        "z_s",
        "vx_s",
        "vy_s",
        "vz_s",
        "m_g",
        "x_g",
        "y_g",
        "z_g",
        "vx_g",
        "vy_g",
        "vz_g",
        "m_dm",
        "x_dm",
        "y_dm",
        "z_dm",
        "vx_dm",
        "vy_dm",
        "vz_dm",
        
    ],
)
def test_same_size_inputs(shorten, random_galaxy_params):
    """Test of inputs lengths."""
    params = random_galaxy_params(stars=10, dm=20, gas=30, seed=42)
    params[shorten] = params[shorten][:-1]
    with pytest.raises(ValueError):
        core.Galaxy(**params)


@pytest.mark.parametrize(
    "pot_name, npart",
    [
        ("pot_s", 10),
        ("pot_g", 20),
        ("pot_dm", 30),
    ],
)
def test_all_potential_inputs(pot_name, npart, random_galaxy_params):
    """Test of potential inputs."""
    seed = 42
    random = np.random.RandomState(seed=seed)
    pot = random.random_sample(size=npart)
    params = random_galaxy_params(stars=10, gas=20, dm=30, seed=42)
    params[pot_name] = pot
    with pytest.raises(ValueError):
        core.Galaxy(**params)


def test_output_galaxy_properties(mock_galaxy):
    """Test of properties and methods outputs."""
    g = mock_galaxy
    g_test = g.angular_momentum()

    assert isinstance(g.kinetic_energy[0], u.Quantity)
    assert isinstance(g.kinetic_energy[1], u.Quantity)
    assert isinstance(g.kinetic_energy[2], u.Quantity)
    assert isinstance(g.potential_energy().pot_dm, u.Quantity)
    assert isinstance(g.potential_energy().pot_s, u.Quantity)
    assert isinstance(g.potential_energy().pot_g, u.Quantity)
    assert isinstance(g.energy[0], u.Quantity)
    assert isinstance(g.energy[1], u.Quantity)
    assert isinstance(g.energy[2], u.Quantity)
    assert isinstance(g_test.J_part, u.Quantity)
    assert isinstance(g_test.Jr_star, u.Quantity)
    assert isinstance(g_test.Jr_part, u.Quantity)
    assert isinstance(g_test.J_star, u.Quantity)
    assert isinstance(g.jcirc().x, u.Quantity)
    assert isinstance(g.jcirc().y, u.Quantity)
    # assert isinstance(g.paramcirc[0], u.Quantity)
    # assert isinstance(g.paramcirc[1], u.Quantity)
    # assert isinstance(g.paramcirc[2], u.Quantity)


def test_energy_method(mock_galaxy):
    """Test energy method."""
    g = mock_galaxy

    E_tot_dm, E_tot_s, E_tot_g = g.energy

    k_s = 0.5 * (g.arr_.vx_s ** 2 + g.arr_.vy_s ** 2 + g.arr_.vz_s ** 2)
    k_dm = 0.5 * (g.arr_.vx_dm ** 2 + g.arr_.vy_dm ** 2 + g.arr_.vz_dm ** 2)
    k_g = 0.5 * (g.arr_.vx_g ** 2 + g.arr_.vy_g ** 2 + g.arr_.vz_g ** 2)

    x = np.hstack((g.arr_.x_s, g.arr_.x_dm, g.arr_.x_g))
    y = np.hstack((g.arr_.y_s, g.arr_.y_dm, g.arr_.y_g))
    z = np.hstack((g.arr_.z_s, g.arr_.z_dm, g.arr_.z_g))
    m = np.hstack((g.arr_.m_s, g.arr_.m_dm, g.arr_.m_g))

    pot = utils.potential(
        da.asarray(x, chunks=100),
        da.asarray(y, chunks=100),
        da.asarray(z, chunks=100),
        da.asarray(m, chunks=100),
    )
    num_s = len(g.arr_.m_s)
    num = len(g.arr_.m_s) + len(g.arr_.m_dm)

    pot_s = pot[:num_s]
    pot_dm = pot[num_s:num]
    pot_g = pot[num:]

    np.testing.assert_allclose(
        E_tot_s.value, k_s - pot_s, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        E_tot_dm.value, k_dm - pot_dm, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        E_tot_g.value, k_g - pot_g, rtol=1e-3, atol=1e-3
    )


def test_energy_method_real_galaxy(mock_real_galaxy):
    """Test energy method with real galaxy."""
    gal = mock_real_galaxy

    E_tot_dm, E_tot_s, E_tot_g = gal.energy

    k_s = 0.5 * (gal.arr_.vx_s ** 2 + gal.arr_.vy_s ** 2 + gal.arr_.vz_s ** 2)
    k_dm = 0.5 * (
        gal.arr_.vx_dm ** 2 + gal.arr_.vy_dm ** 2 + gal.arr_.vz_dm ** 2
    )
    k_g = 0.5 * (gal.arr_.vx_g ** 2 + gal.arr_.vy_g ** 2 + gal.arr_.vz_g ** 2)

    x = np.hstack((gal.arr_.x_s, gal.arr_.x_dm, gal.arr_.x_g))
    y = np.hstack((gal.arr_.y_s, gal.arr_.y_dm, gal.arr_.y_g))
    z = np.hstack((gal.arr_.z_s, gal.arr_.z_dm, gal.arr_.z_g))
    m = np.hstack((gal.arr_.m_s, gal.arr_.m_dm, gal.arr_.m_g))

    pot = utils.potential(
        da.asarray(x, chunks=100),
        da.asarray(y, chunks=100),
        da.asarray(z, chunks=100),
        da.asarray(m, chunks=100),
    )
    num_s = len(gal.arr_.m_s)
    num = len(gal.arr_.m_s) + len(gal.arr_.m_dm)

    pot_s = pot[:num_s]
    pot_dm = pot[num_s:num]
    pot_g = pot[num:]

    np.testing.assert_allclose(
        E_tot_s.value, k_s - pot_s, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        E_tot_dm.value, k_dm - pot_dm, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        E_tot_g.value, k_g - pot_g, rtol=1e-3, atol=1e-3
    )


def test_k_energy(disc_particles_all, halo_particles):
    """Test kinetic energy."""
    (mass_s, pos_s, vel_s, mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles(N_part=100, seed=42)

    k_s = 0.5 * (vel_s[:, 0] ** 2 + vel_s[:, 1] ** 2 + vel_s[:, 2] ** 2)
    k_dm = 0.5 * (vel_dm[:, 0] ** 2 + vel_dm[:, 1] ** 2 + vel_dm[:, 2] ** 2)
    k_g = 0.5 * (vel_g[:, 0] ** 2 + vel_g[:, 1] ** 2 + vel_g[:, 2] ** 2)

    assert (k_s >= 0).all()
    assert (k_dm >= 0).all()
    assert (k_g >= 0).all()


def test_dm_pot_energy(halo_particles):
    """Test potential energy DM."""
    mass_dm, pos_dm, vel_dm = halo_particles(N_part=100, seed=42)

    p_s = utils.potential(
        x=pos_dm[:, 0], y=pos_dm[:, 1], z=pos_dm[:, 2], m=mass_dm
    )
    assert (p_s > 0).all()


def test_stars_and_gas_pot_energy(disc_particles_all):
    """Test potential energy STAR and GAS."""
    (mass_s, pos_s, vel_s, mass_g, pos_g, vel_g) = disc_particles_all

    p_g = utils.potential(
        x=pos_g[:, 0], y=pos_g[:, 1], z=pos_g[:, 2], m=mass_g
    )
    p_s = utils.potential(
        x=pos_s[:, 0], y=pos_s[:, 1], z=pos_s[:, 2], m=mass_s
    )

    assert (p_g > 0).all()
    assert (p_s > 0).all()


def test_total_energy(mock_real_galaxy):
    """Test total energy."""
    g = mock_real_galaxy

    E_tot_dark, E_tot_star, E_tot_gas = g.energy

    (ii_s,) = np.where(E_tot_star.value < 0)
    perc_s = len(ii_s) / len(E_tot_star.value)
    #    (ii_dm,) = np.where(E_tot_dark.value < 0)
    #    perc_dm = len(ii_dm) / len(E_tot_dark.value)
    #    (ii_g,) = np.where(E_tot_gas.value < 0)
    #    perc_g = len(ii_g) / len(E_tot_gas.value)

    assert perc_s > 0.95
    #    assert perc_dm > 0.5
    #    assert perc_g > 0.5
    assert (E_tot_star.value < 0.0).any()
    assert (E_tot_dark.value < 0.0).any()
    assert (E_tot_gas.value < 0.0).any()


def test_type_energy(disc_particles_all, halo_particles):
    """Checks the object."""
    (mass_s, pos_s, vel_s, mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles(N_part=100, seed=42)

    k_s = 0.5 * (vel_s[:, 0] ** 2 + vel_s[:, 1] ** 2 + vel_s[:, 2] ** 2)
    k_dm = 0.5 * (vel_dm[:, 0] ** 2 + vel_dm[:, 1] ** 2 + vel_dm[:, 2] ** 2)
    k_g = 0.5 * (vel_g[:, 0] ** 2 + vel_g[:, 1] ** 2 + vel_g[:, 2] ** 2)

    p_s = utils.potential(
        x=pos_s[:, 0], y=pos_s[:, 1], z=pos_s[:, 2], m=mass_s
    )
    p_dm = utils.potential(
        x=pos_dm[:, 0], y=pos_dm[:, 1], z=pos_dm[:, 2], m=mass_dm
    )
    p_g = utils.potential(
        x=pos_g[:, 0], y=pos_g[:, 1], z=pos_g[:, 2], m=mass_g
    )

    assert isinstance(p_s, (float, np.float64, np.ndarray))
    assert isinstance(p_dm, (float, np.float64, np.ndarray))
    assert isinstance(p_g, (float, np.float64, np.ndarray))
    assert isinstance(k_s, (float, np.float64, np.ndarray))
    assert isinstance(k_dm, (float, np.float64, np.ndarray))
    assert isinstance(k_g, (float, np.float64, np.ndarray))


def test_center_existence(disc_particles_all, halo_particles):
    """Test center existence and uniqueness."""
    (mass_s, pos_s, vel_s, mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles(N_part=100, seed=42)

    gx_c = utils.center(
        pos_s[:, 0],
        pos_s[:, 1],
        pos_s[:, 2],
        pos_dm[:, 0],
        pos_dm[:, 1],
        pos_dm[:, 2],
        pos_g[:, 0],
        pos_g[:, 1],
        pos_g[:, 2],
        mass_s,
        mass_g,
        mass_dm,
    )

    x_gal = np.hstack((gx_c[0], gx_c[3], gx_c[6]))
    y_gal = np.hstack((gx_c[1], gx_c[4], gx_c[7]))
    z_gal = np.hstack((gx_c[2], gx_c[5], gx_c[8]))

    pos_gal = np.vstack((x_gal, y_gal, z_gal))

    assert len(np.where(~pos_gal.any(axis=0))) == 1


def test_angular_momentum_outputs(mock_galaxy):
    """Test object."""
    g = mock_galaxy
    g_test = g.angular_momentum()

    longitude = len(g.x_s) + len(g.x_g) + len(g.x_dm)
    assert np.shape(g_test.J_part.value) == (3, longitude)


def test_jcirc_E_tot_len(mock_galaxy):
    """Check the E_tot array len."""
    g = mock_galaxy

    E_tot_dm, E_tot_s, E_tot_g = g.energy

    E_tot = np.hstack((E_tot_s.value, E_tot_dm.value, E_tot_g.value))
    tot_len = len(E_tot_dm) + len(E_tot_s) + len(E_tot_g)

    assert len(E_tot) == tot_len


def test_jcirc_x_y_len(mock_real_galaxy):
    """Check the x and y array len."""
    gal = mock_real_galaxy

    g_test = gal.jcirc().arr_

    assert len(g_test.x) == len(g_test.y)


def test_param_circ_eps_one_minus_one(mock_real_galaxy):
    """Check is the eps range."""
    gal = mock_real_galaxy

    E_star, eps, eps_r = gal.paramcirc
    (mask,) = np.where(~np.isnan(eps))
    assert (eps[mask] <= 1.0).any()
    assert (eps[mask] >= -1.0).any()


@pytest.mark.parametrize("stars", [(True), (False)])
def test_values_len(stars, mock_real_galaxy):
    """Test the lengths of 2D and 1D array of value mehods."""
    g = mock_real_galaxy

    X, y = g.values(star=stars)

    first = len(g.x_s) if stars else 0

    length = first

    assert X.shape == (length, 10)
    assert y.shape == (length,)
