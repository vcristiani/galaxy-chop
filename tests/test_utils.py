# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities provided by galaxychop."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from galaxychop import core, utils

# =============================================================================
#   POTENTIAL ENERGY
# =============================================================================


def test_Galaxy_potential_energy_already_calculated(galaxy):
    gal = galaxy(seed=42)
    with pytest.raises(ValueError):
        utils.potential(gal)


def test_Galaxy_potential_energy(galaxy):
    gal = galaxy(
        seed=42, stars_potential=False, dm_potential=False, gas_potential=False
    )
    pgal = utils.potential(gal)
    assert isinstance(pgal, core.Galaxy)
    assert np.all(pgal.stars.potential == pgal.potential_energy_[0])
    assert np.all(pgal.dark_matter.potential == pgal.potential_energy_[1])
    assert np.all(pgal.gas.potential == pgal.potential_energy_[2])
