# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
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
def test_JHistogram(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    decomposer = gchop.decomposers.JHistogram()
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
        assert (pset.probabilities is None) or np.isnan(
            pset.probabilities
        ).all()


@pytest.mark.model
def test_JHistogram_spheroid_mirrors_counter_rotating_stars(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.center_and_align(gal, r_cut=30)

    dgal = gchop.decomposers.JHistogram(random_state=42).decompose(gal)

    # Abadi et al. (2003): the spheroid is the counter-rotating stars
    # (eps < 0) plus as many co-rotating ones, mirrored bin by bin, so it
    # holds at most (and, for a disk galaxy, close to) twice as many
    counter_rotating = np.sum(gal.stellar_dynamics().eps < 0)
    spheroid = np.sum(dgal.stars.labels == "Spheroid")

    assert counter_rotating < spheroid <= 2 * counter_rotating
    assert spheroid > 0.95 * 2 * counter_rotating


@pytest.mark.model
def test_JEHistogram(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    decomposer = gchop.decomposers.JEHistogram()
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
        assert (pset.probabilities is None) or np.isnan(
            pset.probabilities
        ).all()
