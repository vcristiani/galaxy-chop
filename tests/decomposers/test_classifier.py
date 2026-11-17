# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2020, 2021, 2022, 2023, 2024, 2025, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import galaxychop as gchop

import numpy as np

import pytest

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RidgeClassifier
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


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.model
def test_ClassifierDecomposer_learns_the_reference(galaxy, reference):
    tree = DecisionTreeClassifier(random_state=0)
    decomposer = gchop.decomposers.ClassifierDecomposer(
        tree, training=[reference]
    )

    # nothing is trained until it is needed
    assert decomposer._trained is None
    assert "training" not in repr(decomposer)

    dgal = decomposer.decompose(galaxy)

    # the given classifier is not modified: a trained copy is used
    with pytest.raises(NotFittedError):
        check_is_fitted(tree)
    assert decomposer.classifier is tree
    assert decomposer.trained_classifier_ is not tree

    assert dgal.method == "ClassifierDecomposer"
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
def test_ClassifierDecomposer_pretrained(galaxy, reference):
    trained = gchop.decomposers.ClassifierDecomposer(
        DecisionTreeClassifier(random_state=0), training=[reference]
    )
    assert trained.fit() is trained
    pretrained = gchop.decomposers.ClassifierDecomposer(
        trained.trained_classifier_
    )

    np.testing.assert_array_equal(
        pretrained.decompose(galaxy).stars.labels,
        trained.decompose(galaxy).stars.labels,
    )


@pytest.mark.model
def test_ClassifierDecomposer_without_probabilities(galaxy, reference):
    decomposer = gchop.decomposers.ClassifierDecomposer(
        RidgeClassifier(), training=[reference]
    )
    dgal = decomposer.decompose(galaxy)

    assert not dgal.has_probabilities
    assert set(dgal.stars.labels) <= {"Disk", "Spheroid", "stars"}


def test_ClassifierDecomposer_not_fitted(galaxy):
    decomposer = gchop.decomposers.ClassifierDecomposer(
        DecisionTreeClassifier()
    )

    with pytest.raises(NotFittedError):
        decomposer.decompose(galaxy)
    with pytest.raises(ValueError, match="no training galaxies"):
        decomposer.fit()


def test_ClassifierDecomposer_training_not_decomposed(galaxy):
    decomposer = gchop.decomposers.ClassifierDecomposer(
        DecisionTreeClassifier(), training=[galaxy]
    )

    with pytest.raises(TypeError, match="must be decomposed"):
        decomposer.fit()
