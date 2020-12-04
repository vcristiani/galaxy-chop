# -*- coding: utf-8 -*-
# This file is part of the Galaxy-Chop Project
# License: MIT

"""Test dynamical decomposition methods."""

# =============================================================================
# IMPORTS
# =============================================================================
from galaxychop import core

import numpy as np

import pytest

from sklearn.cluster import KMeans

# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.xfail
def test_GCKmeans(mock_real_galaxy):
    """ Test GCKmeans."""
    gal = mock_real_galaxy

    gckmeans = core.GCKmeans(n_clusters=5, random_state=0)
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
        core.GCDecomposeMixin(type_values)
