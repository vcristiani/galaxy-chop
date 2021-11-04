# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import galaxychop as gchop

import numpy as np

# =============================================================================
# TESTS
# =============================================================================


def test_KMeans(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    decomposer = gchop.models.KMeans()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        components = decomposer.decompose(gal)

    stars_mask = np.array_equal(components.ptypes, "stars")
    assert np.isfinite(components.labels[stars_mask]).all()
    assert np.isnan(components.labels[~stars_mask]).all()
    assert components.probabilities is None
