# -*- coding: utf-8 -*-
# This file is part of the Galaxy-Chop Project
# License: MIT

"""Test dynamical decomposition methods."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from galaxychop import core

import pytest

from sklearn.cluster import KMeans

# =============================================================================
# TESTS
# =============================================================================


def test_GCKmeans(mock_galaxy):
    """ Test GCKmeans."""
    gal = mock_galaxy

    gckmeans = core.GCKmeans(n_clusters=5, random_state=0)
    result = gckmeans.decompose(gal)
    
    kmeans = KMeans(n_clusters=5, random_state=0)
    X, y = gal.values()
    expected = kmeans.fit_transform(X, y)
    
    np.testing.assert_array_equal(result, expected)