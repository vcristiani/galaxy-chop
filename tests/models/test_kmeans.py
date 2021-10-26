import warnings
import numpy as np

import pytest

import galaxychop as gchop
from galaxychop import models


def test_KMeans(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    decomposer = models.KMeans()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        labels, y = decomposer.decompose(gal)

    stars_mask = np.array_equal(y, "stars")
    assert np.isfinite(labels[stars_mask]).all()
    assert np.isnan(labels[~stars_mask]).all()
