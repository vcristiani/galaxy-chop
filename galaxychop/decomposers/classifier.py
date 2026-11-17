# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2020, 2021, 2022, 2023, 2024, 2025, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Decomposition with a supervised classifier."""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import numpy as np

from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from .core import DecomposedGalaxy, GalaxyDecomposerABC, hparam
from ..utils import doc_inherit

# =============================================================================
# CLASSIFIER DECOMPOSER
# =============================================================================


class ClassifierDecomposer(GalaxyDecomposerABC):
    """
    Decompose galaxies with a supervised scikit-learn classifier.

    The classifier learns the components of the stars of already decomposed
    reference galaxies, from their stellar dynamics, and then predicts them
    for new galaxies.

    Parameters
    ----------
    classifier : sklearn classifier
        Any scikit-learn compatible classifier. With ``training``, a copy of
        it (``sklearn.base.clone``) is trained, so the given classifier is
        never modified. Without ``training`` it must be already trained,
        with the component names (e.g. ``"Disk"``) as classes.
    training : list of DecomposedGalaxy, optional
        Reference galaxies, decomposed, centered and aligned. Their
        classified stars, labeled with their component name, are the
        training set; the stars without a component are left out. The
        training happens on ``fit()``, or on the first ``decompose()`` if
        ``fit()`` was not called.
    attributes : tuple of str, default ("normalized_star_energy", "eps", \
"eps_r")
        Stellar dynamics attributes used as features. They are normalized,
        so they are comparable between galaxies.

    Notes
    -----
    The classifier imitates the decomposition of the reference galaxies:
    trained on the output of another decomposer it reproduces that
    decomposer, only faster. It adds information when the reference labels
    come from somewhere else (e.g. the formation history of the particles in
    a simulation, or an expert classification).

    If the classifier has ``predict_proba()``, the decomposition is
    probabilistic.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> reference = [gchop.decomposers.JHistogram().decompose(g)
    ...              for g in (gal0, gal1)]
    >>> chopper = gchop.decomposers.ClassifierDecomposer(
    ...     RandomForestClassifier(), training=reference)
    >>> chopper.fit()  # optional: decompose() trains it if needed
    >>> chopper.decompose(gal2)

    """

    classifier = attr.ib()
    training = hparam(default=None, repr=False, eq=False)
    attributes = hparam(
        default=("normalized_star_energy", "eps", "eps_r"), converter=tuple
    )

    _trained = attr.ib(init=False, default=None, repr=False, eq=False)

    def fit(self):
        """
        Train a copy of the classifier with the training galaxies.

        Returns
        -------
        ClassifierDecomposer
            The same decomposer, ready to decompose.
        """
        if self.training is None:
            raise ValueError("There are no training galaxies to fit")

        X, labels = [], []
        for galaxy in self.training:
            if not isinstance(galaxy, DecomposedGalaxy):
                raise TypeError(
                    "The training galaxies must be decomposed, "
                    f"found {type(galaxy).__name__}"
                )
            gal_X, y = self._extract_stellar_dynamics_matrix(
                galaxy, self.attributes
            )
            # the stars come first, in the same order as galaxy.stars
            n_stars = len(galaxy.stars)
            valid = self._get_valid_stellar_mask(gal_X, y, self.attributes)
            valid = valid[:n_stars] & np.isfinite(galaxy.stars.components)
            X.append(gal_X[:n_stars][valid])
            labels.append(galaxy.stars.labels[valid])

        trained = clone(self.classifier)
        trained.fit(np.concatenate(X), np.concatenate(labels))
        object.__setattr__(self, "_trained", trained)  # frozen class
        return self

    @property
    def trained_classifier_(self):
        """The trained classifier used to decompose (trains it if needed)."""
        if self._trained is not None:
            return self._trained
        if self.training is not None:
            return self.fit()._trained
        check_is_fitted(self.classifier)
        return self.classifier

    @doc_inherit(GalaxyDecomposerABC.get_attributes)
    def get_attributes(self):
        return list(self.attributes)

    @doc_inherit(GalaxyDecomposerABC.split)
    def split(self, X, y, attributes):
        classifier = self.trained_classifier_
        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(X)
            return probabilities.argmax(axis=1), probabilities

        # classes_ is sorted, so this is the index of each predicted class
        predicted = classifier.predict(X)
        return np.searchsorted(classifier.classes_, predicted), None

    @doc_inherit(GalaxyDecomposerABC.get_component_name_mapping)
    def get_component_name_mapping(self):
        classes = self.trained_classifier_.classes_
        return {i: str(c) for i, c in enumerate(classes)}
