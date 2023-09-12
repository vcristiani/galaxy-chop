# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities  galaxychop.preproc.pcenter"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import preproc

import pandas as pd

# =============================================================================
# TESTS
# =============================================================================


def test_center_and_align(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )

    result = preproc.center_and_align(gal).to_dataframe()
    expected = preproc.star_align(preproc.center(gal)).to_dataframe()

    pd.testing.assert_frame_equal(result, expected)


def test_is_centered_and_aligned(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )

    assert preproc.is_centered_and_aligned(gal) is False

    assert preproc.is_centered_and_aligned(preproc.center(gal)) is False
    assert preproc.is_centered_and_aligned(
        preproc.star_align(preproc.center(gal))
    )
    assert preproc.is_centered_and_aligned(preproc.center_and_align(gal))
