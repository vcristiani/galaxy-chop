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


@pytest.mark.slow
@pytest.mark.model
def test_KMeans(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    decomposer = gchop.models.KMeans(random_state=42)
    dgal = decomposer.decompose(gal)

    assert len(dgal) == len(gal)
    assert len(dgal.stars) == len(gal.stars)
    assert len(dgal.dark_matter) == len(gal.dark_matter)
    assert len(dgal.gas) == len(gal.gas)

    total_labels_no_nans = pd.notna(dgal.stars.labels).sum()
    assert total_labels_no_nans <= len(gal.stars)

    total_labels_nans = pd.isna(dgal.stars.labels).sum()
    assert total_labels_nans == len(gal.stars) - total_labels_no_nans

    assert np.all(dgal.dark_matter.labels == "dark_matter")
    assert np.all(dgal.gas.labels == "gas")

    for pset in (dgal.stars, dgal.dark_matter, dgal.gas):
        assert (
            (pset.probabilities is None) or np.isnan(pset.probabilities).all()
        )
