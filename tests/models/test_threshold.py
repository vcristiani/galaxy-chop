# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import galaxychop as gchop

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_JThreshold(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    decomposer = gchop.models.JThreshold()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
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


@pytest.mark.parametrize("eps_cut", [(1.1), (-1.1)])
def test_JThreshold_eps_cut_value_error(eps_cut):
    with pytest.raises(ValueError):
        gchop.models.JThreshold(eps_cut=eps_cut)
