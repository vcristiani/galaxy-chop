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

def save_data(N_part=100):
    m, pos, vel = solid_disk(N_part)
    data = np.ndarray([len(m),4])
    data[:,0] = pos[:,0]
    data[:,1] = pos[:,1]
    data[:,2] = pos[:,2]
    data[:,3] = m

    np.savetxt('test_data/mock_particles.dat', data, fmt='%12.6f')

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def disc_particles():
    m, pos, vel = solid_disk(N_part=100)
    return pos[:,0], pos[:,1], pos[:,2], m

#def read_data():
#    fpotential = np.loadtxt('test_data/fpotential_test.dat')

#    return fpotential
# =============================================================================
# TESTS
# =============================================================================


def test_daskpotential(disc_particles):
    dpotential = utils.potential(*disc_particles)
    #fpotential = read_data()
    fpotential = np.loadtxt('test_data/fpotential_test.dat')

    np.testing.assert_allclose(dpotential, fpotential, rtol=1e-4, atol=1e-3)
    


