# -*- coding: utf-8 -*-

# This file is part of the Galaxy-Chop Project
# License: MIT

# =============================================================================
# IMPORTS
# =============================================================================

import pytest

import numpy as np


import sys
sys.path.insert(0, '.')
from galaxychop import utils

# =============================================================================
# Random state
# =============================================================================
# Fix the random state
random = np.random.RandomState(seed=31)


# =============================================================================
# Defining utility functions for mocking data
# =============================================================================
def solid_disc(N_part=100, rmax=30, rmin=5, omega=10):
    """
    Creates a set of particles that belong to a rigid body rotating disc,
    sampling particles from a flat annulus, with maximum radius and minimum 
    radius `(rmax, rmin)`.
    
    The angular velocity `omega` is used to set an angular rotation around 
    `z` axis. The function delivers mass vector (set to identical unity mass),

    """
    r = (rmax - rmin) * np.random.random_sample(size=N_part) + rmin
    phi0 = 2 * np.pi * np.random.random_sample(size=N_part)
    mass = 1. * np.ones_like(r)

    x = r * np.cos(phi0)
    y = r * np.sin(phi0)
    z = 1 * np.random.random_sample(size=N_part) - 0.5

    xdot = -1 * omega * r * np.sin(phi0)
    ydot = omega * r * np.cos(phi0)
    zdot = np.zeros_like(xdot)

    pos = np.array([x, y, z]).T
    vel = np.array([xdot, ydot, zdot]).T
    return mass, pos, vel

# transformation matrix
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
    A = np.array([
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
    A = np.array([
        [np.cos(theta), 0, -1 * np.sin(theta)],
        [0, 1,  0],
        [np.sin(theta), 0, np.cos(theta)],
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
    A = np.array([
        
        [np.cos(theta),  -1 * np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
        ]
    )
    return A

# rotation function
def rotate(pos, vel, matrix):

    pos_rot = pos @ matrix
    vel_rot = vel @ matrix
    
    return pos_rot, vel_rot


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def disc_zero_angle():
    m, pos, vel = solid_disc(N_part=1000)

    return m, pos, vel

@pytest.fixture
def disc_xrotation():
    m, pos, vel = solid_disc(N_part=1000)
    a = rot_matrix_xaxis(theta=0.5* np.pi * np.random.random())

    return m, pos @ a, vel @ a, a


# =============================================================================
# TESTS
# =============================================================================

# rotate it 
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

    # we want this to be the identity
    invtest = a @ np.linalg.inv(gxchA)
    np.testing.assert_allclose(invtest, np.identity(3), atol=1e-3, rtol=1e-4)
