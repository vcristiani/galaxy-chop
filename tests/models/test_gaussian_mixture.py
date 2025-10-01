# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================


import galaxychop as gchop

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.slow
@pytest.mark.model
def test_GaussianMixture(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    decomposer = gchop.models.GaussianMixture(random_state=42, n_init=1)
    dgal = decomposer.decompose(gal)

    assert len(dgal) == len(gal)
    assert len(dgal.stars) == len(gal.stars)
    assert len(dgal.dark_matter) == len(gal.dark_matter)
    assert len(dgal.gas) == len(gal.gas)

    total_labels_no_nans = pd.notna(dgal.stars.labels).sum()
    assert total_labels_no_nans <= len(gal.stars)

    total_labels_nans = pd.isna(dgal.stars.labels).sum()
    assert total_labels_nans == len(gal.stars) - total_labels_no_nans

    dm_labels = pd.Series(dgal.dark_matter.labels)
    gas_labels = pd.Series(dgal.gas.labels)
    assert (pd.isna(dm_labels).all()) or ((dm_labels == "dark_matter").all())
    assert (pd.isna(gas_labels).all()) or ((gas_labels == "gas").all())

    assert dgal.stars.probabilities.shape == (
        len(gal.stars),
        decomposer.n_components,
    )


@pytest.mark.slow
@pytest.mark.model
def test_AutoGaussianMixture(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    decomposer = gchop.models.AutoGaussianMixture(random_state=42, n_init=1)
    dgal = decomposer.decompose(gal)

    assert len(dgal) == len(gal)
    assert len(dgal.stars) == len(gal.stars)
    assert len(dgal.dark_matter) == len(gal.dark_matter)
    assert len(dgal.gas) == len(gal.gas)

    total_labels_no_nans = pd.notna(dgal.stars.labels).sum()
    assert total_labels_no_nans <= len(gal.stars)

    total_labels_nans = pd.isna(dgal.stars.labels).sum()
    assert total_labels_nans == len(gal.stars) - total_labels_no_nans

    dm_labels = pd.Series(dgal.dark_matter.labels)
    gas_labels = pd.Series(dgal.gas.labels)
    assert (pd.isna(dm_labels).all()) or ((dm_labels == "dark_matter").all())
    assert (pd.isna(gas_labels).all()) or ((gas_labels == "gas").all())

    assert dgal.stars.probabilities.shape == (len(gal.stars), 4)
