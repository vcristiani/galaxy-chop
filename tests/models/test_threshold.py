# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import galaxychop as gchop

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.model
def test_JThreshold(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.star_align(gchop.preproc.center(gal))

    decomposer = gchop.models.JThreshold()

    components = decomposer.decompose(gal)

    assert len(components) == len(gal)
    assert len(gal.stars) == np.sum(components.ptypes == "stars")
    assert len(gal.dark_matter) == np.sum(components.ptypes == "dark_matter")
    assert len(gal.gas) == np.sum(components.ptypes == "gas")

    # the total number of no nans must be <= the number of stars
    total_labels_no_nans = np.isfinite(components.labels).sum()
    assert total_labels_no_nans <= len(gal.stars)

    # the nans must be the subtraction between stars and no_nans + dm + gas
    total_labels_nans = np.isnan(components.labels).sum()
    assert total_labels_nans == (
        len(gal.stars)
        - total_labels_no_nans
        + len(gal.dark_matter)
        + len(gal.gas)
    )

    assert components.probabilities is None


@pytest.mark.model
@pytest.mark.parametrize("eps_cut", [(1.1), (-1.1)])
def test_JThreshold_eps_cut_value_error(eps_cut):
    with pytest.raises(ValueError):
        gchop.models.JThreshold(eps_cut=eps_cut)
