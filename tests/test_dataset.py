# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# ======================================================================
# IMPORTS
# ======================================================================

import galaxychop as gc

# =============================================================================
# TESTS
# =============================================================================


def test_load_dark():
    """Check the dark.dat array len."""
    result = gc.dataset.load_dark()

    assert result.shape == (350, 8)


def test_load_gas():
    """Check the gas.dat array len."""
    result = gc.dataset.load_gas()

    assert result.shape == (250, 8)


def test_load_star():
    """Check the star.dat array len."""
    result = gc.dataset.load_star()

    assert result.shape == (300, 8)


def test_load_dark_TNG_394242():
    """Check the dark_ID_394242.npy array len."""
    result = gc.dataset.load_dark_394242()

    assert result.shape == (21156, 8)


def test_load_gas_TNG_394242():
    """Check the gas_ID_394242.npy array len."""
    result = gc.dataset.load_gas_394242()

    assert result.shape == (4061, 8)


def test_load_star_TNG_394242():
    """Check the star_ID_394242.npy array len."""
    result = gc.dataset.load_star_394242()

    assert result.shape == (32067, 8)


def test_load_pot_dark_TNG_394242():
    """Check the potential_dark_ID_394242.npy array len."""
    result = gc.dataset.load_pot_dark_394242()

    assert result.shape == (21156,)


def test_load_pot_gas_TNG_394242():
    """Check the potential_gas_ID_394242.npy array len."""
    result = gc.dataset.load_pot_gas_394242()

    assert result.shape == (4061,)


def test_load_pot_star_TNG_394242():
    """Check the potential_star_ID_394242.npy array len."""
    result = gc.dataset.load_pot_star_394242()

    assert result.shape == (32067,)
