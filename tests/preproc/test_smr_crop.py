# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Test utilities  galaxychop.preproc.smr_crop"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop.preproc import smr_crop

import numpy as np


# =============================================================================
# TESTS
# =============================================================================


def test_half_star_mass_radius_crop(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    agal = smr_crop.half_star_mass_radius_crop(gal, num_radii=1)

    gal_m_s = gal.stars.m.sum().to_value()
    agal_m_s = agal.stars.m.sum().to_value()

    assert np.isclose(
        agal_m_s, (gal_m_s / 2.0), rtol=1e-05, atol=1e-08, equal_nan=False
    )
