# -*- coding: utf-8 -*-

# This file is part of the Galaxy-Chop Project
# License: MIT

# =============================================================================
# IMPORTS
# =============================================================================

import pytest

import numpy as np

from numpy.testing import assert_equal, assert_

# =============================================================================
#  FIXTURES
# =============================================================================
def solid_disc(N_part=100, rmax=30, rmin=5, omega=10):
    r = (rmax - rmin) * np.random.random_sample(size=N_part) + rmin
    phi0 = 2 * np.pi * np.random.random_sample(size=N_part)
    mass = 1. * np.ones_like(r)

    x = r * np.cos(phi0)
    y = r * np.sin(phi0)
    z = 2 * np.random.random_sample(size=N_part)

    xdot = omega * r * np.sin(phi0)
    ydot = omega * r * np.sin(phi0)
    zdot = np.zeros_like(xdot)

    pos = np.array([x, y, z])
    vel = np.array([xdot, ydot, zdot])
    return mass, pos.T, vel.T

def rot_matrix_xaxis(theta=0):
    A = np.array(
        [1, 0, 0],
        [0, np.cos(theta),  -1 * np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ]
    )
    return A

def rotation(pos, vel, matrix):

    pos_rot = np.dot(A, pos.T)
    vel_rot = np.dot(A, vel.T)
    
    return pos_rot, vel_rot

# =============================================================================
# TESTS
# =============================================================================
class BaseTest(object):

    def __init__(self):
        
        self.m = 
        self.pos = 
        self.vel = 
        self.r_corte = 

    def test_aling():
        assert(True)

    def test_dens_2D():
    ...

    def test_dens_3D():
    ...
    