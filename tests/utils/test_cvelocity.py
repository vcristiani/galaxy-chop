# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""test for galaxychop.utils.cvelocity"""


# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import constants
from galaxychop.utils import cvelocity

import numpy as np


# =============================================================================
# TESTS
# =============================================================================


def test_circular_velocity():
    mass = np.array([1.0, 2.0, 3.0])
    radius = np.array([1.0, 2.0, 4.0])

    result = cvelocity.circular_velocity(mass, radius)

    enclosed = np.array([1.0, 1.0 + 2.0, 1.0 + 2.0 + 3.0])
    expected = np.sqrt(constants.G * enclosed / radius)
    np.testing.assert_allclose(result, expected)


def test_circular_velocity_keeps_input_order():
    mass = np.array([3.0, 1.0, 2.0])
    radius = np.array([4.0, 1.0, 2.0])

    result = cvelocity.circular_velocity(mass, radius)

    # same particles as above, unsorted: each keeps its own value
    enclosed = np.array([6.0, 1.0, 3.0])
    expected = np.sqrt(constants.G * enclosed / radius)
    np.testing.assert_allclose(result, expected)


def test_circular_velocity_zero_radius_is_nan():
    mass = np.array([1.0, 2.0])
    radius = np.array([0.0, 1.0])

    result = cvelocity.circular_velocity(mass, radius)

    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1], np.sqrt(constants.G * 3.0 / 1.0))


def test_circular_velocity_length_mismatch_is_nan():
    result = cvelocity.circular_velocity(np.ones(2), np.ones(3))

    assert result.shape == (3,)
    assert np.isnan(result).all()
