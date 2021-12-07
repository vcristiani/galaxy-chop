# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
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


@pytest.mark.slow
@pytest.mark.model
def test_GaussianMixture(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    decomposer = gchop.models.GaussianMixture(random_state=42, n_init=1)

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

    # shape of the probs must be the size of the galaxy, number of components
    assert np.shape(components.probabilities) == (
        len(gal),
        decomposer.n_components,
    )

    # the total number of no nans must be <= the number of stars
    total_probs_no_nans = (
        np.isfinite(components.probabilities).all(axis=1).sum()
    )
    assert total_probs_no_nans <= len(gal.stars)

    # the nans must be the subtraction between stars and no_nans + dm + gas
    total_probs_nans = np.isnan(components.probabilities).any(axis=1).sum()
    assert total_probs_nans == (
        len(gal.stars)
        - total_probs_no_nans
        + len(gal.dark_matter)
        + len(gal.gas)
    )


@pytest.mark.slow
@pytest.mark.model
def test_AutoGaussianMixture(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    decomposer = gchop.models.AutoGaussianMixture(random_state=42, n_init=1)

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

    # shape of the probs must be the size of the galaxy, and 4
    assert np.shape(components.probabilities) == (len(gal), 4)

    # the total number of no nans must be <= the number of stars
    total_probs_no_nans = (
        np.isfinite(components.probabilities).any(axis=1).sum()
    )
    assert total_probs_no_nans <= len(gal.stars)

    # the nans must be the subtraction between stars and no_nans + dm + gas
    total_probs_nans = np.isnan(components.probabilities).any(axis=1).sum()
    assert total_probs_nans == (
        len(gal.stars)
        - total_probs_no_nans
        + len(gal.dark_matter)
        + len(gal.gas)
    )
