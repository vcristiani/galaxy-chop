# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test dynamical decomposition methods."""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import core, models

import numpy as np

import pytest

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# =============================================================================
# MIXIN PROTOCOL
# =============================================================================


def get_subclasses(cls, add_cls=False):
    clss = set()
    if cls.__module__.startswith("galaxychop.models"):
        for scls in cls.__subclasses__():
            clss.update(get_subclasses(scls, add_cls=True))
        if add_cls:
            clss.add(cls)
    return clss


@pytest.mark.parametrize("model", get_subclasses(models.GalaxyDecomposeMixin))
def test_GalaxyDecomposeMixin_protocol(mock_real_galaxy, model):
    chopper = model()

    # dirty "X" feature matrix to check, get_clean_mask and add_dirty
    dirty_X = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, np.nan, 9],  # the row 0 is not clean
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ]
    )

    # get_clean_mask
    mask = chopper.get_clean_mask(dirty_X)
    np.testing.assert_array_equal(mask, [1, 2])

    # add_dirty
    labels = chopper.label_dirty(dirty_X, [3, 3], [1, 2])
    np.testing.assert_array_equal(labels, [-1, 3, 3])

    # get_columns
    columns = chopper.get_columns()
    assert isinstance(columns, (list, tuple))  # is a list or tuple
    assert len(columns) == len(set(columns))  # no repeated columns
    for column in columns:  # all columns are from core.Column?
        core.Columns(column)

    # decompose
    result = chopper.decompose(mock_real_galaxy)
    assert chopper is result


# =============================================================================
# MODELS
# =============================================================================


def test_JHistogram_len(mock_real_galaxy):
    """Test the lengths of labels."""
    gal = mock_real_galaxy
    X, y = gal.values()
    chopper = models.JHistogram(seed=10)
    chopper.decompose(gal)

    longitude = len(chopper.labels_)
    assert np.shape(X) == (longitude, 10)


def test_JHistogram_outputs(mock_real_galaxy):
    """Test outputs of JHistogram model."""
    gal = mock_real_galaxy
    chopper = models.JHistogram(seed=10)
    chopper.decompose(gal)

    labels = chopper.labels_
    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)
    (comp_nan,) = np.where(labels == -1)
    len_lab = len(labels[comp0]) + len(labels[comp1]) + len(labels[comp_nan])

    assert (labels >= -1).all() and (labels <= 1).all()
    assert len_lab == len(labels)


def test_JHistogram_histogram(mock_real_galaxy):
    """Test the number of particles per bin."""
    gal = mock_real_galaxy
    X, y = gal.values()
    chopper = models.JHistogram(seed=10)
    chopper.decompose(gal)
    labels = chopper.labels_
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

    chop = models.JThreshold()
    chop.decompose(gal)

    longitude = len(chop.labels_)
    assert np.shape(X) == (longitude, 10)


def test_GCChop_outputs(mock_real_galaxy):
    """Test outputs of JThreshold model."""
    gal = mock_real_galaxy
    chop = models.JThreshold()
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
    chop = models.JThreshold()
    chop.decompose(gal)
    labels = chop.labels_

    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)
    (comp_nan,) = np.where(labels == -1)

    X, y = gal.values()

    expected_esf = np.where((X[:, 8] <= 0.6) & (~np.isnan(X[:, 8])))[0]
    expected_disk = np.where((X[:, 8] > 0.6) & (~np.isnan(X[:, 8])))[0]
    expected_nan = np.where(np.isnan(X[:, 8]))[0]

    np.testing.assert_array_equal(comp0, expected_esf)
    np.testing.assert_array_equal(comp1, expected_disk)
    np.testing.assert_array_equal(comp_nan, expected_nan)


@pytest.mark.parametrize("eps_cut", [(1.1), (-1.1)])
def test_GCChop_eps_cut_value_error(eps_cut):
    with pytest.raises(ValueError):
        models.JThreshold(eps_cut)


def test_JEHistogram_len(mock_real_galaxy):
    """Test the lengths of labels."""
    gal = mock_real_galaxy
    X, y = gal.values()
    cristiani = models.JEHistogram(seed=10)
    cristiani.decompose(gal)

    longitude = len(cristiani.labels_)
    assert np.shape(X) == (longitude, 10)


def test_JEHistogram_outputs(mock_real_galaxy):
    """Test outputs of JEHistogram model."""
    gal = mock_real_galaxy
    cristiani = models.JEHistogram(seed=10)
    cristiani.decompose(gal)

    labels = cristiani.labels_
    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)
    (comp_nan,) = np.where(labels == -1)
    len_lab = len(labels[comp0]) + len(labels[comp1]) + len(labels[comp_nan])

    assert (labels >= -1).all() and (labels <= 1).all()
    assert len_lab == len(labels)


def test_JEHistogram_histogram(mock_real_galaxy):
    """Test the number of particles per bin."""
    gal = mock_real_galaxy
    X, y = gal.values()
    cristiani = models.JEHistogram(seed=10)
    cristiani.decompose(gal)
    labels = cristiani.labels_
    (comp0,) = np.where(labels == 0)
    (comp1,) = np.where(labels == 1)

    full_histogram = np.histogram(X[:, 8], bins=100, range=(-1.0, 1.0))
    comp0_histogram = np.histogram(X[:, 8][comp0], bins=100, range=(-1.0, 1.0))
    comp1_histogram = np.histogram(X[:, 8][comp1], bins=100, range=(-1.0, 1.0))

    comp0_hist_plus_comp1_hist = comp0_histogram[0] + comp1_histogram[0]

    np.testing.assert_equal(comp0_hist_plus_comp1_hist, full_histogram[0])


def test_Kmeans(mock_real_galaxy):
    """Test KMeans."""
    gal = mock_real_galaxy

    chopper = models.KMeans(n_clusters=5, random_state=0)
    result = chopper.decompose(gal)
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
    """Test GaussianMixture."""
    gal = mock_real_galaxy

    chopper = models.GaussianMixture(n_components=5, random_state=0)
    result = chopper.decompose(gal)
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

    autogmm = models.AutoGaussianMixture(c_bic=0.1)
    autogmm.decompose(gal)

    predict_proba = autogmm.probability_of_gaussianmixture
    probability = autogmm.probability
    labels = autogmm.labels_

    sum_predict_proba = np.apply_along_axis(sum, 1, predict_proba)
    sum_probability = np.apply_along_axis(sum, 1, probability)
    sum_expected = np.ones(len(labels[np.where(labels != -1)]))

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

    autogmm = models.AutoGaussianMixture(c_bic=0.1)
    autogmm.decompose(gal)

    labels = autogmm.labels_
    print(labels)

    assert (labels < 4).all()
