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

import numpy as np

import pytest

from sklearn.cluster import KMeans

# =============================================================================
# TESTS
# =============================================================================


def test_GCKmeans(mock_galaxy):
    """ Test GCKmeans."""
    gal = mock_galaxy

    gckmeans = models.GCKmeans(n_clusters=5, random_state=0)
    result = gckmeans.decompose(gal)

    kmeans = KMeans(n_clusters=5, random_state=0)
    X, y = gal.values()
    expected = kmeans.fit_transform(X, y)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "type_values", [("string", np.random.rand(1), np.inf, np.nan)]
)
def test_type_error_GCDecomposeMixin_class(type_values):
    with pytest.raises(TypeError):
        models.GCDecomposeMixin(type_values)



