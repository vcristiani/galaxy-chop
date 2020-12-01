# -*- coding: utf-8 -*-
# This file is part of the Galaxy-Chop Project
# License: MIT

"""Test input data."""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import utils

import numpy as np

import pytest

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
