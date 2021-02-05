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
from sklearn.mixture import GaussianMixture

# =============================================================================
# TESTS
# =============================================================================


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
        models.GCDecomposeMixin(type_values)


def test_GCAbadi_len(mock_real_galaxy):
    """Test the lengths of labels."""
    gal = mock_real_galaxy
    X, y = gal.values()
    abadi = models.GCAbadi(seed=10)
    abadi.decompose(gal)

    longitude = len(abadi.labels_)
    assert np.shape(X) == (longitude, 10)


def test_GCAbadi_outputs(mock_real_galaxy):
    """Test outputs of GCAbadi model."""
    gal = mock_real_galaxy
    abadi = models.GCAbadi(seed=10)
    abadi.decompose(gal)

    labels = abadi.labels_
    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)
    (comp_nan,) = np.where(labels == -1)
    len_lab = len(labels[comp0]) + len(labels[comp1]) + len(labels[comp_nan])

    assert (labels >= -1).all() and (labels <= 1).all()
    assert len_lab == len(labels)


def test_GCAbadi_histogram(mock_real_galaxy):
    """Test the number of particles per bin."""
    gal = mock_real_galaxy
    X, y = gal.values()
    abadi = models.GCAbadi(seed=10)
    abadi.decompose(gal)
    labels = abadi.labels_
    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)

    full_histogram = np.histogram(X[:, 8], bins=100, range=(-1.0, 1.0))
    comp0_histogram = np.histogram(X[:, 8][comp0], bins=100, range=(-1.0, 1.0))
    comp1_histogram = np.histogram(X[:, 8][comp1], bins=100, range=(-1.0, 1.0))

    comp0_hist_plus_comp1_hist = comp0_histogram[0] + comp1_histogram[0]

    np.testing.assert_equal(comp0_hist_plus_comp1_hist, full_histogram[0])


def test_GCChop_len(mock_real_galaxy):
    """Test the lengths of labels."""
    gal = mock_real_galaxy
    X, y = gal.values()

    chop = models.GCChop()
    chop.decompose(gal)

    longitude = len(chop.labels_)
    assert np.shape(X) == (longitude, 10)


def test_GCChop_outputs(mock_real_galaxy):
    """Test outputs of GCChop model."""
    gal = mock_real_galaxy
    chop = models.GCChop()
    chop.decompose(gal)
    labels = chop.labels_

    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)
    (comp_nan,) = np.where(labels == -1)

    len_lab = len(labels[comp0]) + len(labels[comp1]) + len(labels[comp_nan])

    assert (labels >= -1).all() and (labels <= 1).all()
    assert len_lab == len(labels)


def test_GCChop_eps_cut(mock_real_galaxy):
    """Tests the number of particles in each component."""
    gal = mock_real_galaxy
    chop = models.GCChop()
    chop.decompose(gal)
    labels = chop.labels_

    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)
    (comp_nan,) = np.where(labels == -1)

    X, y = gal.values()

    clean_eps = ~np.isnan(X[:, 8])
    expected_esf = np.where(X[clean_eps, 8] <= 0.6)[0]
    expected_disk = np.where(X[clean_eps, 8] > 0.6)[0]
    expected_nan = np.where(np.isnan(X[:, 8]))[0]

    np.testing.assert_array_equal(comp0, expected_esf)
    np.testing.assert_array_equal(comp1, expected_disk)
    np.testing.assert_array_equal(comp_nan, expected_nan)


@pytest.mark.parametrize("eps_cut", [(1.1), (-1.1)])
def test_GCChop_eps_cut_value_error(eps_cut):
    with pytest.raises(ValueError):
        models.GCChop(eps_cut)


def test_GCCristiani_len(mock_real_galaxy):
    """Test the lengths of labels."""
    gal = mock_real_galaxy
    X, y = gal.values()
    je = models.GCJE(seed=10)
    je.decompose(gal)

    longitude = len(je.labels_)
    assert np.shape(X) == (longitude, 10)


def test_GCCristiani_outputs(mock_real_galaxy):
    """Test outputs of GCCristiani model."""
    gal = mock_real_galaxy
    je = models.GCJE(seed=10)
    je.decompose(gal)

    labels = je.labels_
    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)
    (comp_nan,) = np.where(labels == -1)
    len_lab = len(labels[comp0]) + len(labels[comp1]) + len(labels[comp_nan])

    assert (labels >= -1).all() and (labels <= 1).all()
    assert len_lab == len(labels)


def test_GCCristiani_histogram(mock_real_galaxy):
    """Test the number of particles per bin."""
    gal = mock_real_galaxy
    X, y = gal.values()
    je = models.GCJE(seed=10)
    je.decompose(gal)
    labels = je.labels_
    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)

    full_histogram = np.histogram(X[:, 8], bins=100, range=(-1.0, 1.0))
    comp0_histogram = np.histogram(X[:, 8][comp0], bins=100, range=(-1.0, 1.0))
    comp1_histogram = np.histogram(X[:, 8][comp1], bins=100, range=(-1.0, 1.0))

    comp0_hist_plus_comp1_hist = comp0_histogram[0] + comp1_histogram[0]

    np.testing.assert_equal(comp0_hist_plus_comp1_hist, full_histogram[0])


def test_GCKmeans(mock_real_galaxy):
    """Test GCKmeans."""
    gal = mock_real_galaxy

    gckmeans = models.GCKmeans(n_clusters=5, random_state=0)
    result = gckmeans.decompose(gal)
    (clean_label_gal,) = np.where(result.labels_ != -1)

    kmeans = KMeans(n_clusters=5, random_state=0)
    X, y = gal.values()
    (clean_eps,) = np.where(~np.isnan(X[:, 8]))
    expected = kmeans.fit(X[:, [7, 8, 9]][clean_eps], y[clean_eps])

    np.testing.assert_array_equal(
        result.labels_[clean_label_gal], expected.labels_
    )

    np.testing.assert_allclose(
        result.cluster_centers_,
        expected.cluster_centers_,
        rtol=1e-7,
        atol=1e-8,
    )


def test_GCGmm(mock_real_galaxy):
    """Test GCGmm."""
    gal = mock_real_galaxy

    gcgmm = models.GCGmm(n_components=5, random_state=0)
    result = gcgmm.decompose(gal)
    (clean_label_gal,) = np.where(result.labels_ != -1)

    gmm = GaussianMixture(n_components=5, random_state=0)
    X, y = gal.values()
    (clean_eps,) = np.where(~np.isnan(X[:, 8]))
    expected = gmm.fit(X[:, [7, 8, 9]][clean_eps], y[clean_eps])

    np.testing.assert_array_equal(
        result.labels_[clean_label_gal],
        expected.predict(X[:, [7, 8, 9]][clean_eps]),
    )

    np.testing.assert_allclose(
        result.means_,
        expected.means_,
        rtol=1e-7,
        atol=1e-8,
    )


def test_GCAutogmm_prob(mock_real_galaxy):
    """Test that the probabilities obtained by the method sum to 1."""
    gal = mock_real_galaxy

    autogmm = models.GCAutogmm(c_bic=0.1)
    autogmm.decompose(gal)

    predict_proba = autogmm.probability_of_gaussianmixture
    probability = autogmm.probability
    labels = autogmm.labels_

    sum_predict_proba = np.apply_along_axis(sum, 1, predict_proba)
    sum_probability = np.apply_along_axis(sum, 1, probability)
    sum_expected = np.ones(len(labels))

    np.testing.assert_allclose(
        sum_predict_proba,
        sum_expected,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        sum_probability,
        sum_expected,
        rtol=1e-8,
        atol=1e-8,
    )


def test_GCAutogmm_label(mock_real_galaxy):
    """Test of label values."""
    gal = mock_real_galaxy

    autogmm = models.GCAutogmm(c_bic=0.1)
    autogmm.decompose(gal)

    labels = autogmm.labels_
    print(labels)

    assert (labels < 4).all()
