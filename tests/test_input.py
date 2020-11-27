# -*- coding: utf-8 -*-
# This file is part of the Galaxy-Chop Project
# License: MIT

"""Test imput data."""

# =============================================================================
# IMPORTS
# =============================================================================

from os import path

import astropy.units as u

import dask.array as da

from galaxychop import core
from galaxychop import utils

import numpy as np

import pytest

# =============================================================================
# Random state
# =============================================================================

random = np.random.RandomState(seed=42)

# =============================================================================
# Defining utility functions for mocking data
# =============================================================================


def rot_matrix_xaxis(theta=0):
    """
    Rotation matrix of a transformation around X axis.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians

    Returns
    -------
    A : `np.ndarray`
        Rotation matrix, with shape (3, 3)
    """
    A = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -1 * np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return A


def rot_matrix_yaxis(theta=0):
    """
    Rotation matrix of a transformation around Y axis.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians
    Returns
    -------
    A : `np.ndarray`
        Rotation matrix, with shape (3, 3)
    """
    A = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-1 * np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return A


def rot_matrix_zaxis(theta=0):
    """
    Rotation matrix of a transformation around Z axis.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians
    Returns
    -------
    A : `np.ndarray`
        Rotation matrix, with shape (3, 3)
    """
    A = np.array(
        [
            [np.cos(theta), -1 * np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return A


def rotate(pos, vel, matrix):
    """
    Rotate.

    Apply the rotation `matrix` to a set of particles positions `pos` and
    velocities `vel`

    Parameters
    ----------
    pos : `np.ndarray`, shape = N_part, 3
        Positions of particles
    vel : `np.ndarray`, shape = N_part, 3
        Velocities of particles
    matrix : `np.ndarray`
        Rotation matrix, with shape (3, 3)

    Returns
    -------
    pos_rot : `np.ndarray`, shape = N_part, 3
        Rotated, positions of particles
    vel_rot : `np.ndarray`, shape = N_part, 3
        Rotated, velocities of particles
    """
    pos_rot = pos @ matrix
    vel_rot = vel @ matrix

    return pos_rot, vel_rot


def save_data(N_part=100):
    """
    Save data.

    This function saves a file with mock particles in a solid disk created with
    `solid_disk` function to run potentials with `potential_test.f90`
    to validate the potential function with dask

    Parameters
    ----------
    N_part : `int`
        The total number of particles to obtain

    Returns
    -------
    File named  `mock_particles.dat` on the folder tests/test_data
    with 4 columns and N_part rows. From left to right:
    x, y, z : Positions
    mass : Masses
    """
    mass, pos, vel = solid_disk(N_part)
    data = np.ndarray([len(mass), 4])
    data[:, 0] = pos[:, 0]
    data[:, 1] = pos[:, 1]
    data[:, 2] = pos[:, 2]
    data[:, 3] = mass

    np.savetxt("test_data/mock_particles.dat", data, fmt="%12.6f")


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def galaxy_params():
    """
    Galaxy parameter for test.

    This return a function of a dictionary with random params of a Galaxy
    object
    """

    def make(seed, stars, gas, dm):

        random = np.random.default_rng(seed=seed)

        x_s = random.random(stars)
        y_s = random.random(stars)
        z_s = random.random(stars)
        vx_s = random.random(stars)
        vy_s = random.random(stars)
        vz_s = random.random(stars)
        m_s = random.random(stars)

        x_g = random.random(gas)
        y_g = random.random(gas)
        z_g = random.random(gas)
        vx_g = random.random(gas)
        vy_g = random.random(gas)
        vz_g = random.random(gas)
        m_g = random.random(gas)

        x_dm = random.random(dm)
        y_dm = random.random(dm)
        z_dm = random.random(dm)
        vx_dm = random.random(dm)
        vy_dm = random.random(dm)
        vz_dm = random.random(dm)
        m_dm = random.random(dm)

        params = {
            "x_s": x_s,
            "y_s": y_s,
            "z_s": z_s,
            "vx_s": vx_s,
            "vy_s": vy_s,
            "vz_s": vz_s,
            "m_s": m_s,
            "x_dm": x_dm,
            "y_dm": y_dm,
            "z_dm": z_dm,
            "vx_dm": vx_dm,
            "vy_dm": vy_dm,
            "vz_dm": vz_dm,
            "m_dm": m_dm,
            "x_g": x_g,
            "y_g": y_g,
            "z_g": z_g,
            "vx_g": vx_g,
            "vy_g": vy_g,
            "vz_g": vz_g,
            "m_g": m_g,
        }
        return params

    return make


@pytest.fixture(scope="session")
def solid_disk():
    """Mock solid disk.

    Creates a mock solid disc of particles with masses
    and velocities.
    """

    def make(N_part=100, rmax=30, rmin=2, omega=10, seed=42):

        random = np.random.RandomState(seed=seed)

        r = (rmax - rmin) * random.random_sample(size=N_part) + rmin
        phi0 = 2 * np.pi * random.random_sample(size=N_part)
        mass = 1.0e8 * np.ones_like(r)

        x = r * np.cos(phi0)
        y = r * np.sin(phi0)
        z = 1 * random.random_sample(size=N_part) - 0.5

        xdot = -1 * omega * r * np.sin(phi0)
        ydot = omega * r * np.cos(phi0)
        zdot = np.zeros_like(xdot)

        pos = np.array([x, y, z]).T
        vel = np.array([xdot, ydot, zdot]).T

        return mass, pos, vel

    return make


@pytest.fixture(scope="session")
def mock_dm_halo():
    """
    Mock Dark Matter Halo.

    Creates a mock DM Halo of particles with masses
    and velocities.
    """

    def make(N_part=1000, rmax=100, seed=55):

        random = np.random.RandomState(seed=seed)

        r = random.random_sample(size=N_part) * rmax

        cos_t = random.random_sample(size=N_part) * 2.0 - 1
        phi0 = 2 * np.pi * random.random_sample(size=N_part)
        sin_t = np.sqrt(1 - cos_t ** 2)
        mass = 1.0e10 * np.ones_like(r)

        x = r * sin_t * np.cos(phi0)
        y = r * sin_t * np.sin(phi0)
        z = r * cos_t

        pos = np.array([x, y, z]).T

        return mass, pos

    return make


@pytest.fixture
def disc_zero_angle(solid_disk):
    """Disc with no angle of inclination."""
    mass, pos, vel = solid_disk(N_part=1000)
    return mass, pos, vel


@pytest.fixture
def disc_xrotation(solid_disk):
    """Disc rotated over x axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_xaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture
def disc_yrotation(solid_disk):
    """Disc rotated over y axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_yaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture
def disc_zrotation(solid_disk):
    """Disc rotated over z axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_zaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture
def disc_particles(solid_disk):
    """Solid disc without velocities."""
    mass, pos, vel = solid_disk(N_part=100)
    return pos[:, 0], pos[:, 1], pos[:, 2], mass


@pytest.fixture
def disc_particles_all(solid_disk):
    """Solid disc wit velocities."""
    mass_s, pos_s, vel_s = solid_disk(N_part=100)
    mass_g, pos_g, vel_g = solid_disk(N_part=100)

    return mass_s, pos_s, vel_s, mass_g, pos_g, vel_g


@pytest.fixture
def halo_particles(mock_dm_halo):
    """Spherical mock halo."""
    mass_dm, pos_dm = mock_dm_halo(N_part=100)
    vel_dm = random.random_sample(size=(100, 3))

    return mass_dm, pos_dm, vel_dm


@pytest.fixture
def mock_galaxy(disc_particles_all, halo_particles):
    """Mock galaxy."""
    (mass_s, pos_s, vel_s, mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles

    g = core.Galaxy(
        x_s=pos_s[:, 0] * u.kpc,
        y_s=pos_s[:, 1] * u.kpc,
        z_s=pos_s[:, 2] * u.kpc,
        vx_s=vel_s[:, 0] * (u.km / u.s),
        vy_s=vel_s[:, 1] * (u.km / u.s),
        vz_s=vel_s[:, 2] * (u.km / u.s),
        m_s=mass_s * u.M_sun,
        x_dm=pos_dm[:, 0] * u.kpc,
        y_dm=pos_dm[:, 1] * u.kpc,
        z_dm=pos_dm[:, 2] * u.kpc,
        vx_dm=vel_dm[:, 0] * (u.km / u.s),
        vy_dm=vel_dm[:, 1] * (u.km / u.s),
        vz_dm=vel_dm[:, 2] * (u.km / u.s),
        m_dm=mass_dm * u.M_sun,
        x_g=pos_g[:, 0] * u.kpc,
        y_g=pos_g[:, 1] * u.kpc,
        z_g=pos_g[:, 2] * u.kpc,
        vx_g=vel_g[:, 0] * (u.km / u.s),
        vy_g=vel_g[:, 1] * (u.km / u.s),
        vz_g=vel_g[:, 2] * (u.km / u.s),
        m_g=mass_g * u.M_sun,
    )

    return g


@pytest.fixture
def mock_real_galaxy():
    """Mock real galaxy."""
    dm = np.loadtxt(path.abspath(path.curdir) + "/legacy/dark.dat")
    s = np.loadtxt(path.abspath(path.curdir) + "/legacy/star.dat")
    g = np.loadtxt(path.abspath(path.curdir) + "/legacy/gas_.dat")
    gal = core.Galaxy(
        x_s=s[:, 1] * u.kpc,
        y_s=s[:, 2] * u.kpc,
        z_s=s[:, 3] * u.kpc,
        vx_s=s[:, 4] * (u.km / u.s),
        vy_s=s[:, 5] * (u.km / u.s),
        vz_s=s[:, 6] * (u.km / u.s),
        m_s=s[:, 0] * 1e10 * u.M_sun,
        x_dm=dm[:, 1] * u.kpc,
        y_dm=dm[:, 2] * u.kpc,
        z_dm=dm[:, 3] * u.kpc,
        vx_dm=dm[:, 4] * (u.km / u.s),
        vy_dm=dm[:, 5] * (u.km / u.s),
        vz_dm=dm[:, 6] * (u.km / u.s),
        m_dm=dm[:, 0] * 1e10 * u.M_sun,
        x_g=g[:, 1] * u.kpc,
        y_g=g[:, 2] * u.kpc,
        z_g=g[:, 3] * u.kpc,
        vx_g=g[:, 4] * (u.km / u.s),
        vy_g=g[:, 5] * (u.km / u.s),
        vz_g=g[:, 6] * (u.km / u.s),
        m_g=g[:, 0] * 1e10 * u.M_sun,
    )

    return gal


# =============================================================================
# TESTS
# =============================================================================
param_list = [
    "x_s",
    "y_s",
    "z_s",
    "vx_s",
    "vy_s",
    "vz_s",
    "m_s",
    "x_g",
    "y_g",
    "z_g",
    "vx_g",
    "vy_g",
    "vz_g",
    "m_g",
    "x_dm",
    "y_dm",
    "z_dm",
    "vx_dm",
    "vy_dm",
    "vz_dm",
    "m_dm",
]


@pytest.mark.parametrize("shorten", param_list)
def test_same_size_inputs(shorten, galaxy_params):
    """Test of inputs lengths."""
    params = galaxy_params(seed=42, stars=10, gas=20, dm=30)
    params[shorten] = params[shorten][:-1]
    with pytest.raises(ValueError):
        core.Galaxy(**params)


def test_getrotmat0(disc_zero_angle):
    """Test rotation matrix 1."""
    gxchA = utils._get_rot_matrix(*disc_zero_angle)

    np.testing.assert_allclose(1.0, gxchA[2, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 1], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[1, 2], rtol=1e-4, atol=1e-3)


def test_invert_xaxis(disc_xrotation):
    """Test rotation matrix 2."""
    m, pos, vel, a = disc_xrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(1.0, gxchA[0, 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 2], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[1, 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 0], rtol=1e-3, atol=1e-3)


def test_invert_yaxis(disc_yrotation):
    """Test rotation matrix 3."""
    m, pos, vel, a = disc_yrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(0.0, gxchA[0, 0], rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(1.0, gxchA[0, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 2], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[1, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 1], rtol=1e-3, atol=1e-3)


def test_invert_zaxis(disc_zrotation):
    """Test rotation matrix 4."""
    m, pos, vel, a = disc_zrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(1.0, gxchA[2, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 1], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[1, 2], rtol=1e-4, atol=1e-3)


@pytest.mark.xfail
def test_daskpotential(disc_particles):
    """Test potential function."""
    dpotential = utils.potential(*disc_particles)
    fpotential = np.loadtxt("tests/test_data/fpotential_test.dat")
    np.testing.assert_allclose(dpotential, fpotential, rtol=1e-4, atol=1e-3)


def test_output_galaxy_properties(mock_galaxy):
    """Test output of properties."""
    g = mock_galaxy
    g_test = g.angular_momentum()

    assert isinstance(g.energy[0], u.Quantity)
    assert isinstance(g.energy[1], u.Quantity)
    assert isinstance(g.energy[2], u.Quantity)
    assert isinstance(g_test.J_part, u.Quantity)
    assert isinstance(g_test.Jr_star, u.Quantity)
    assert isinstance(g_test.Jr, u.Quantity)
    assert isinstance(g_test.J_star, u.Quantity)
    assert isinstance(g.jcirc().x, u.Quantity)
    assert isinstance(g.jcirc().y, u.Quantity)
    assert isinstance(g.paramcirc[0], u.Quantity)
    assert isinstance(g.paramcirc[1], u.Quantity)
    assert isinstance(g.paramcirc[2], u.Quantity)


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
    """Test energy method."""
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

    mass_dm, pos_dm, vel_dm = halo_particles

    k_s = 0.5 * (vel_s[:, 0] ** 2 + vel_s[:, 1] ** 2 + vel_s[:, 2] ** 2)
    k_dm = 0.5 * (vel_dm[:, 0] ** 2 + vel_dm[:, 1] ** 2 + vel_dm[:, 2] ** 2)
    k_g = 0.5 * (vel_g[:, 0] ** 2 + vel_g[:, 1] ** 2 + vel_g[:, 2] ** 2)

    assert (k_s >= 0).all()
    assert (k_dm >= 0).all()
    assert (k_g >= 0).all()


def test_dm_pot_energy(halo_particles):
    """Test potential energy DM."""
    mass_dm, pos_dm, vel_dm = halo_particles

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

    mass_dm, pos_dm, vel_dm = halo_particles

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

    assert isinstance(p_s, (float, np.float, np.ndarray))
    assert isinstance(p_dm, (float, np.float, np.ndarray))
    assert isinstance(p_g, (float, np.float, np.ndarray))
    assert isinstance(k_s, (float, np.float, np.ndarray))
    assert isinstance(k_dm, (float, np.float, np.ndarray))
    assert isinstance(k_g, (float, np.float, np.ndarray))


def test_center_existence(disc_particles_all, halo_particles):
    """Test center existence and uniqueness."""
    (mass_s, pos_s, vel_s, mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles

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
    assert (eps <= 1.0).any()
    assert (eps >= -1.0).any()


@pytest.mark.parametrize(
    "stars_number, gas_number, dm_number, stars , gas, dm",
    [
        (1, 1, 1, True, True, True),
        (10, 20, 30, True, False, False),
        (300, 183, 2934, False, True, False),
        (30, 18, 293, False, False, True),
        (3, 83, 24, True, True, False),
        (28, 43, 94, False, True, True),
        (382, 8321, 834, True, False, True),
        (32, 821, 84, False, False, False),
    ],
)
def test_values_len(
    stars_number, gas_number, dm_number, stars, gas, dm, galaxy_params
):
    """Test the lengths of 2D and 1D array of value mehods."""
    params = galaxy_params(
        seed=42, stars=stars_number, gas=gas_number, dm=dm_number
    )
    g = core.Galaxy(**params)

    X, y = g.values(star=stars, gas=gas, dm=dm)

    first = stars_number if stars else 0
    second = gas_number if gas else 0
    third = dm_number if dm else 0

    length = first + second + third

    assert X.shape == (length, 7)
    assert y.shape == (length,)
