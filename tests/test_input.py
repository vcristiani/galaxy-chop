# -*- coding: utf-8 -*-

# This file is part of the Galaxy-Chop Project
# License: MIT

# =============================================================================
# IMPORTS
# =============================================================================

import pytest

import numpy as np

from galaxychop import utils

# =============================================================================
# Random state
# =============================================================================
# Fix the random state
random = np.random.RandomState(seed=42)

# =============================================================================
# Defining utility functions for mocking data
# =============================================================================


def solid_disk(N_part=100, rmax=30, rmin=5, omega=10):

    """
    Creates a set of particles that belong to a rigid body rotating disk,
    sampling particles from a flat annulus, with maximum radius and minimum
    radius `(rmax, rmin)` and thickness iqual to 1.

    The angular velocity `omega` is used to set an angular rotation around
    `z` axis. The function delivers mass vector (set to identical unity mass),
    the positions and velocities as `(N_part, 3)` arrays.

    Parameters
    ----------
    N_part : `int`
        The total number of particles to obtain
    rmax : `float`
        The maximum radius of the disk
    rmin : `float`
        The minimum radius of the disk
    omega : `float`
        Angular velocity of the

    Returns
    -------
    mass : `np.ndarray`, shape = N_part, 1
        Masses per particle, identically 1 for all.
    pos : `np.ndarray`, shape = N_part, 3
        Positions of particles
    vel : `np.ndarray`, shape = N_part, 3
        Velocities of particles
    """

    r = (rmax - rmin) * random.random_sample(size=N_part) + rmin
    phi0 = 2 * np.pi * random.random_sample(size=N_part)
    mass = 1. * np.ones_like(r)

    x = r * np.cos(phi0)
    y = r * np.sin(phi0)
    z = 1 * random.random_sample(size=N_part) - 0.5

    xdot = -1 * omega * r * np.sin(phi0)
    ydot = omega * r * np.cos(phi0)
    zdot = np.zeros_like(xdot)

    pos = np.array([x, y, z]).T
    vel = np.array([xdot, ydot, zdot]).T

    return mass, pos, vel


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
            [0, np.cos(theta),  -1 * np.sin(theta)],
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
            [0, 1,  0],
            [-1*np.sin(theta), 0, np.cos(theta)],
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
            [np.cos(theta),  -1 * np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return A


def rotate(pos, vel, matrix):
    """
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


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def disc_zero_angle():
    m, pos, vel = solid_disk(N_part=1000)

    return m, pos, vel


@pytest.fixture
def disc_xrotation():
    m, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_xaxis(theta=0.3 * np.pi * random.random())

    return m, pos @ a, vel @ a, a


@pytest.fixture
def disc_yrotation():
    m, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_yaxis(theta=0.3 * np.pi * random.random())

    return m, pos @ a, vel @ a, a


@pytest.fixture
def disc_zrotation():
    m, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_zaxis(theta=0.3 * np.pi * random.random())

    return m, pos @ a, vel @ a, a

# =============================================================================
# TESTS
# =============================================================================


def test_getrotmat0(disc_zero_angle):
    gxchA = utils._get_rot_matrix(*disc_zero_angle)

    np.testing.assert_allclose(1., gxchA[2, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[2, 1], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[2, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[0, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[1, 2], rtol=1e-4, atol=1e-3)


def test_invert_xaxis(disc_xrotation):
    m, pos, vel, a = disc_xrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(1., gxchA[0, 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[0, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[0, 2], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[1, 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[2, 0], rtol=1e-3, atol=1e-3)


def test_invert_yaxis(disc_yrotation):
    m, pos, vel, a = disc_yrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(0., gxchA[0, 0], rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(1., gxchA[0, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[0, 2], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[1, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[2, 1], rtol=1e-3, atol=1e-3)


def test_invert_zaxis(disc_zrotation):
    m, pos, vel, a = disc_zrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(1., gxchA[2, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[2, 1], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[2, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[0, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0., gxchA[1, 2], rtol=1e-4, atol=1e-3)
