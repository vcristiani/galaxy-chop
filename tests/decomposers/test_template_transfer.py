# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2020, 2021, 2022, 2023, 2024, 2025, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import pickle
import warnings
from unittest import mock

import attr

import galaxychop as gchop

import numpy as np

import pytest

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    return gchop.preproc.center_and_align(gal)


@pytest.fixture(scope="module")
def reference(galaxy):
    return gchop.decomposers.JThreshold().decompose(galaxy)


@pytest.fixture(scope="module")
def jhistogram_reference(galaxy):
    return gchop.decomposers.JHistogram().decompose(galaxy)


class SpyTree(DecisionTreeClassifier):
    """A decision tree that keeps the sample_weight given to fit()."""

    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = sample_weight
        return super().fit(X, y, sample_weight=sample_weight)


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.model
def test_TemplateTransfer_learns_the_reference(galaxy, reference):
    tree = DecisionTreeClassifier(random_state=0)
    decomposer = gchop.decomposers.TemplateTransfer(tree)

    # nothing is trained until fit() is called
    assert decomposer._trained is None

    assert decomposer.fit([reference]) is decomposer
    dgal = decomposer.decompose(galaxy)

    # the given classifier is not modified: a trained copy is used
    with pytest.raises(NotFittedError):
        check_is_fitted(tree)
    assert decomposer.classifier is tree
    assert decomposer.trained_classifier_ is not tree

    assert dgal.method == "TemplateTransfer"
    assert dgal.component_name_mapping == {0: "Disk", 1: "Spheroid"}
    assert dgal.has_probabilities

    # JThreshold is a cut on eps, which a tree learns exactly
    classified = np.isfinite(reference.stars.components)
    np.testing.assert_array_equal(
        dgal.stars.labels[classified], reference.stars.labels[classified]
    )
    assert np.all(dgal.dark_matter.labels == "dark_matter")
    assert np.all(dgal.gas.labels == "gas")


@pytest.mark.model
def test_TemplateTransfer_pretrained(galaxy, reference):
    # a single galaxy is accepted too
    trained = gchop.decomposers.TemplateTransfer(
        DecisionTreeClassifier(random_state=0)
    ).fit(reference)
    pretrained = gchop.decomposers.TemplateTransfer(
        trained.trained_classifier_
    )

    np.testing.assert_array_equal(
        pretrained.decompose(galaxy).stars.labels,
        trained.decompose(galaxy).stars.labels,
    )


@pytest.mark.model
def test_TemplateTransfer_without_probabilities(galaxy, reference):
    decomposer = gchop.decomposers.TemplateTransfer(RidgeClassifier()).fit(
        [reference]
    )
    dgal = decomposer.decompose(galaxy)

    assert not dgal.has_probabilities
    assert set(dgal.stars.labels) <= {"Disk", "Spheroid", "stars"}


def test_TemplateTransfer_no_training_galaxies():
    decomposer = gchop.decomposers.TemplateTransfer(DecisionTreeClassifier())

    with pytest.raises(ValueError, match="no training galaxies"):
        decomposer.fit([])


def test_TemplateTransfer_training_not_decomposed(galaxy):
    decomposer = gchop.decomposers.TemplateTransfer(DecisionTreeClassifier())

    with pytest.raises(TypeError, match="must be decomposed"):
        decomposer.fit([galaxy])


# MASS WEIGHTS ================================================================


@pytest.mark.model
def test_TemplateTransfer_mass_weighted(reference):
    decomposer = gchop.decomposers.TemplateTransfer(SpyTree(random_state=0))
    assert decomposer.mass_weighted
    sample_weight = decomposer.fit(
        [reference]
    ).trained_classifier_.sample_weight_

    # the masses (plain Msun values) of the classified stars
    classified = np.isfinite(reference.stars.components)
    expected = reference.stars.m.to_value("Msun")[classified]
    assert isinstance(sample_weight, np.ndarray)
    assert len(np.unique(expected)) > 1  # the masses are not uniform
    np.testing.assert_array_equal(sample_weight, expected)


@pytest.mark.model
def test_TemplateTransfer_not_mass_weighted(reference):
    decomposer = gchop.decomposers.TemplateTransfer(
        SpyTree(random_state=0), mass_weighted=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decomposer.fit([reference])

    assert decomposer.trained_classifier_.sample_weight_ is None


@pytest.mark.model
def test_TemplateTransfer_mass_weighted_unsupported(reference):
    decomposer = gchop.decomposers.TemplateTransfer(KNeighborsClassifier())
    with pytest.warns(UserWarning, match="does not accept sample_weight"):
        decomposer.fit([reference])
    check_is_fitted(decomposer.trained_classifier_)

    unweighted = gchop.decomposers.TemplateTransfer(
        KNeighborsClassifier(), mass_weighted=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unweighted.fit([reference])


# TRAINING LABELS =============================================================


def test_TemplateTransfer_different_components(reference):
    other = attr.evolve(
        reference,
        method="AutoGaussianMixture",
        component_name_mapping={
            0: "Halo",
            1: "Bulge",
            2: "Cold disk",
            3: "Warm disk",
        },
    )
    decomposer = gchop.decomposers.TemplateTransfer(DecisionTreeClassifier())

    with pytest.raises(ValueError, match="training galaxy 2") as err:
        decomposer.fit([reference, reference, other])
    assert "['Bulge', 'Cold disk', 'Halo', 'Warm disk']" in str(err.value)
    assert "['Disk', 'Spheroid']" in str(err.value)


@pytest.mark.model
def test_TemplateTransfer_different_methods(reference, jhistogram_reference):
    decomposer = gchop.decomposers.TemplateTransfer(
        DecisionTreeClassifier(random_state=0)
    )

    with pytest.warns(UserWarning, match="different methods") as record:
        decomposer.fit([reference, jhistogram_reference])
    assert "['JHistogram', 'JThreshold']" in str(record[0].message)
    check_is_fitted(decomposer.trained_classifier_)


# EXPLICIT TRAINING AND PARALLELISM ===========================================


def test_TemplateTransfer_training_requires_fit(galaxy):
    decomposer = gchop.decomposers.TemplateTransfer(SpyTree())

    # decompose() never trains: fit() must be called explicitly
    with mock.patch.object(
        gchop.decomposers.TemplateTransfer,
        "fit",
        side_effect=AssertionError("trained implicitly"),
    ):
        with pytest.raises(NotFittedError, match="call fit"):
            decomposer.decompose(galaxy)
        with pytest.raises(NotFittedError, match="call fit"):
            decomposer.trained_classifier_

    assert decomposer._trained is None


@pytest.mark.model
def test_TemplateTransfer_pickle_keeps_training(galaxy, reference):
    decomposer = gchop.decomposers.TemplateTransfer(
        SpyTree(random_state=0)
    ).fit([reference])

    restored = pickle.loads(pickle.dumps(decomposer))
    assert restored._trained is not None

    # decompose() must use the restored classifier, without training again
    with (
        mock.patch.object(
            SpyTree, "fit", side_effect=AssertionError("trained again")
        ),
        mock.patch.object(
            gchop.decomposers.TemplateTransfer,
            "fit",
            side_effect=AssertionError("trained again"),
        ),
    ):
        dgal = restored.decompose(galaxy)

    np.testing.assert_array_equal(
        dgal.stars.labels, decomposer.decompose(galaxy).stars.labels
    )
