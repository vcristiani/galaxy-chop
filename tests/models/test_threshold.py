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

import pandas as pd

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.model
def test_JThreshold(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    decomposer = gchop.models.JThreshold()
    dgal = decomposer.decompose(gal)

    assert len(dgal) == len(gal)
    assert len(dgal.stars) == len(gal.stars)
    assert len(dgal.dark_matter) == len(gal.dark_matter)
    assert len(dgal.gas) == len(gal.gas)

    total_labels_no_nans = pd.notna(dgal.stars.labels).sum()
    assert total_labels_no_nans <= len(gal.stars)

    total_labels_nans = pd.isna(dgal.stars.labels).sum()
    assert total_labels_nans == len(gal.stars) - total_labels_no_nans

    assert (dgal.dark_matter.labels == "dark_matter").all()
    assert (dgal.gas.labels == "gas").all()

    assert (
        dgal.stars.probabilities is None
        or np.isnan(dgal.stars.probabilities).all()
    )
    assert (
        dgal.dark_matter.probabilities is None
        or np.isnan(dgal.dark_matter.probabilities).all()
    )
    assert (
        dgal.gas.probabilities is None
        or np.isnan(dgal.gas.probabilities).all()
    )


@pytest.mark.model
@pytest.mark.parametrize("eps_cut", [1.1, -1.1])
def test_JThreshold_eps_cut_value_error(eps_cut):
    with pytest.raises(ValueError):
        gchop.models.JThreshold(eps_cut=eps_cut)
