# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test dynamical decomposition methods."""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import models
from galaxychop import sklearn_models

import numpy as np

import pytest

from sklearn.cluster import KMeans

# =============================================================================
# TESTS
# =============================================================================


def test_GCKmeans(mock_real_galaxy):
    """Test GCKmeans."""
    gal = mock_real_galaxy

    gckmeans = sklearn_models.GCKmeans(n_clusters=5, random_state=0)
    result = gckmeans.decompose(gal)

    kmeans = KMeans(n_clusters=5, random_state=0)
    X, y = gal.values()
    expected = kmeans.fit_transform(X, y)

    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-06)


@pytest.mark.parametrize(
    "type_values",
    [
        "string",
        np.random.rand(1),
        np.inf,
        np.nan,
    ],
)
def test_type_error_GCDecomposeMixin_class(type_values):
    """Test type error GCDecomposeMixin."""
    with pytest.raises(TypeError):
        sklearn_models.GCDecomposeMixin(type_values)


def test_GCAbadi_len(mock_real_galaxy):
    """Test the lengths of labels."""
    gal = mock_real_galaxy
    X, y = gal.values()
    abadi = models.GCAbadi()
    abadi.decompose(gal)

    longitude = len(abadi.labels_)
    assert np.shape(X) == (longitude, 3)


def test_GCAbadi_outputs(mock_real_galaxy):
    """Test output of GCAbadi model."""
    gal = mock_real_galaxy
    abadi = models.GCAbadi()
    abadi.decompose(gal)

    assert (abadi.labels_ >= 0).all() and (abadi.labels_ <= 1).all()
