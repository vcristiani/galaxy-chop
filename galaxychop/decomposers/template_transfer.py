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

import warnings

import attr

import numpy as np

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from .core import DecomposedGalaxy, GalaxyDecomposerABC, hparam
from ..utils import doc_inherit

# =============================================================================
# TEMPLATE TRANSFER
# =============================================================================


class TemplateTransfer(GalaxyDecomposerABC):
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
        training only happens when ``fit()`` is called, and it must be
        called before ``decompose()``.
    attributes : tuple of str, default ("normalized_star_energy", "eps", \
"eps_r")
        Stellar dynamics attributes used as features. They are normalized,
        so they are comparable between galaxies.
    mass_weighted : bool, default True
        Weight each training star by its mass (``sample_weight``), so the
        classifier learns the components by mass, as ``total_mass()``
        reports them, and not by number of particles. It matters when the
        stellar particles have different masses (e.g. TNG). If the
        classifier does not accept ``sample_weight`` (e.g.
        ``KNeighborsClassifier``), a ``UserWarning`` is emitted and it is
        trained without weights.

    Notes
    -----
    The classifier imitates the decomposition of the reference galaxies:
    trained on the output of another decomposer it reproduces that
    decomposer, only faster. It adds information when the reference labels
    come from somewhere else (e.g. the formation history of the particles in
    a simulation, or an expert classification).

    If the classifier has ``predict_proba()``, the decomposition is
    probabilistic.

    All the training galaxies must have the same component names (e.g. all
    ``{"Spheroid", "Disk"}``), otherwise ``fit()`` raises a ``ValueError``.
    If they share the names but come from different methods (e.g.
    ``JThreshold`` and ``JHistogram``), a ``UserWarning`` is emitted, since
    labels of different methods are being mixed.

    The training is never implicit: with ``training``, ``decompose()``
    raises a ``NotFittedError`` until ``fit()`` is called. A trained
    decomposer can be pickled with its trained classifier, so it can be
    trained once and then used to decompose several galaxies in parallel
    (e.g. with joblib or multiprocessing).

    Do not evaluate the decomposer on a galaxy that is also in ``training``:
    the agreement with the reference is inflated (data leakage). To evaluate
    it, split the reference by galaxy, not by particle.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> reference = [gchop.decomposers.JHistogram().decompose(g)
    ...              for g in (gal0, gal1)]
    >>> chopper = gchop.decomposers.TemplateTransfer(
    ...     RandomForestClassifier(), training=reference).fit()
    >>> chopper.decompose(gal2)

    """

    classifier = attr.ib()
    training = hparam(default=None, repr=False, eq=False)
    attributes = hparam(
        default=("normalized_star_energy", "eps", "eps_r"), converter=tuple
    )
    mass_weighted = hparam(default=True, converter=bool)

    _trained = attr.ib(init=False, default=None, repr=False, eq=False)

    def fit(self):
        """
        Train a copy of the classifier with the training galaxies.

        It must be called before ``decompose()`` when the decomposer has
        ``training``.

        Returns
        -------
        TemplateTransfer
            The same decomposer instance (``self``), trained and ready to
            decompose, so it can be chained:
            ``TemplateTransfer(clf, training=refs).fit().decompose(gal)``.
        """
        if self.training is None or len(self.training) == 0:
            raise ValueError("There are no training galaxies to fit")

        self._check_training_labels()

        X, labels, masses = [], [], []
        for galaxy in self.training:
            gal_X, y = self._extract_stellar_dynamics_matrix(
                galaxy, self.attributes
            )
            # the stars come first, in the same order as galaxy.stars
            n_stars = len(galaxy.stars)
            valid = self._get_valid_stellar_mask(gal_X, y, self.attributes)
            valid = valid[:n_stars] & np.isfinite(galaxy.stars.components)
            X.append(gal_X[:n_stars][valid])
            labels.append(galaxy.stars.labels[valid])
            masses.append(galaxy.stars.arr_.m[valid])  # plain Msun values

        trained = clone(self.classifier)
        fit_kws = {}
        if self.mass_weighted:
            if has_fit_parameter(trained, "sample_weight"):
                fit_kws["sample_weight"] = np.concatenate(masses)
            else:
                warnings.warn(
                    f"{type(trained).__name__}.fit() does not accept "
                    "sample_weight, so it is trained without the stellar "
                    "masses as weights. Use mass_weighted=False to silence "
                    "this warning.",
                    UserWarning,
                    stacklevel=2,
                )

        trained.fit(np.concatenate(X), np.concatenate(labels), **fit_kws)
        object.__setattr__(self, "_trained", trained)  # frozen class
        return self

    def _check_training_labels(self):
        """Check that the training galaxies share their component names."""
        for galaxy in self.training:
            if not isinstance(galaxy, DecomposedGalaxy):
                raise TypeError(
                    "The training galaxies must be decomposed, "
                    f"found {type(galaxy).__name__}"
                )

        first = self.training[0]
        names = set(first.component_name_mapping.values())
        for idx, galaxy in enumerate(self.training[1:], start=1):
            other = set(galaxy.component_name_mapping.values())
            if other != names:
                raise ValueError(
                    f"The training galaxy {idx} has the components "
                    f"{sorted(other)}, but the galaxy 0 has {sorted(names)}. "
                    "All the training galaxies must have the same components"
                )

        methods = {galaxy.method for galaxy in self.training}
        if len(methods) > 1:
            warnings.warn(
                "The training galaxies were decomposed with different "
                f"methods {sorted(methods)}: their labels are being mixed",
                UserWarning,
                stacklevel=3,
            )

    @property
    def trained_classifier_(self):
        """The trained classifier used to decompose.

        Raises ``NotFittedError`` if the decomposer has ``training`` and
        ``fit()`` was not called, or if the given classifier is not trained.
        """
        if self._trained is not None:
            return self._trained
        if self.training is not None:
            raise NotFittedError(
                f"This {type(self).__name__} has training galaxies but it "
                "is not trained yet: call fit() before decompose()"
            )
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
