# -*- coding: utf-8 -*-
# This file is part of the Galaxy-Chop Project
# License: MIT

"""Test imput data."""

# =============================================================================
# IMPORTS
# =============================================================================

import pytest
import numpy as np
from galaxychop import utils

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


@pytest.fixture(scope="session")
def solid_disk():
    """Mock solid disk.

    Creates a mock solid disc of particles with masses
    and velocities.
    """

    def make(N_part=100, rmax=30, rmin=5, omega=10, seed=42):

        random = np.random.RandomState(seed=seed)

        r = (rmax - rmin) * random.random_sample(size=N_part) + rmin
        phi0 = 2 * np.pi * random.random_sample(size=N_part)
        mass = 1.0 * np.ones_like(r)

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
def disc_zero_angle(solid_disk):
    """Disc with no angle of inclination."""
    mass, pos, vel = solid_disk(N_part=1000)
    return mass, pos, vel


@pytest.fixture(scope="session")
def disc_xrotation(solid_disk):
    """Disc rotated over x axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_xaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture(scope="session")
def disc_yrotation(solid_disk):
    """Disc rotated over y axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_yaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture(scope="session")
def disc_zrotation(solid_disk):
    """Disc rotated over z axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_zaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture(scope="session")
def disc_particles(solid_disk):
    """Solid disc without velocities."""
    mass, pos, vel = solid_disk(N_part=100)
    return pos[:, 0], pos[:, 1], pos[:, 2], mass


# =============================================================================
# TESTS
# =============================================================================


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
