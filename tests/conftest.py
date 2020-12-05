# -*- coding: utf-8 -*-
# This file is part of the Galaxy-Chop Project
# License: MIT

"""Fixtures input data."""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import astropy.units as u

from galaxychop import core

import numpy as np

import pytest


# =============================================================================
# PATHS
# =============================================================================

PATH = Path(os.path.abspath(os.path.dirname(__file__)))

TEST_DATA_PATH = PATH / "tests" / "test_data"

TEST_DATA_REAL_PATH = TEST_DATA_PATH / "real"

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
def random_galaxy_params():
    """
    Galaxy parameter for test.

    This return a function of a dictionary with random params of a Galaxy
    object
    """

    def make(stars, gas, dm, seed):

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
    random = np.random.RandomState(seed=42)
    a = rot_matrix_xaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture
def disc_yrotation(solid_disk):
    """Disc rotated over y axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    random = np.random.RandomState(seed=42)
    a = rot_matrix_yaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture
def disc_zrotation(solid_disk):
    """Disc rotated over z axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    random = np.random.RandomState(seed=42)
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


@pytest.fixture(scope="session")
def halo_particles(mock_dm_halo):
    """Spherical mock halo."""

    def make(N_part=100, seed=None):
        random = np.random.RandomState(seed=seed)
        mass_dm, pos_dm = mock_dm_halo(N_part=N_part)
        vel_dm = random.random_sample(size=(N_part, 3))

        return mass_dm, pos_dm, vel_dm

    return make


@pytest.fixture
def mock_galaxy(disc_particles_all, halo_particles):
    """Mock galaxy."""
    (mass_s, pos_s, vel_s, mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles(N_part=100, seed=42)

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

    dm = np.loadtxt(TEST_DATA_REAL_PATH / "dark.dat")
    s = np.loadtxt(TEST_DATA_REAL_PATH / "star.dat")
    g = np.loadtxt(TEST_DATA_REAL_PATH / "gas_.dat")
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
