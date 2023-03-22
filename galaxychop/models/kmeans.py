# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


"""Module models."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from sklearn import cluster

from ._base import DynamicStarsDecomposerMixin, GalaxyDecomposerABC, hparam
from ..utils import doc_inherit

# =============================================================================
# KNN
# =============================================================================


class KMeans(DynamicStarsDecomposerMixin, GalaxyDecomposerABC):
    """KMeans class.

    Implementation of Scikit-learn [6]_ K-means as a method for dynamically
    decomposing galaxies.

    Parameters
    ----------
    n_components : int, default=2
        The number of clusters to form as well as the number of centroids to
        generate.
    init : {'k-means++', 'random'}, callable or array-like of shape
    (n_clusters, n_features), default="k-means++"
        Parameter of :py:class:``k-Means`` class into ``scikit-learn`` library.
    n_init : int, default=10
        Parameter of :py:class:``k-Means`` class into ``scikit-learn`` library.
    max_iter : int, default=300
        Parameter of :py:class:``k-Means`` class into ``scikit-learn`` library.
    tol : float, default=0.0001
        Parameter of :py:class:``k-Means`` class into ``scikit-learn`` library.
    verbose : int, default=0
        Parameter of :py:class:``k-Means`` class into ``scikit-learn`` library.
    random_state : int, default=None
        Parameter of :py:class:``k-Means`` class into ``scikit-learn`` library.
    algorithm : {“lloyd”, “elkan”}, default="auto"
        Parameter of :py:class:``k-Means`` class into ``scikit-learn`` library.

    Notes
    -----
    More information for `KMeans` class:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Examples
    --------
    Example of implementation of KMeans Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.read_hdf5(...)
    >>> galaxy = gchop.utils.star_align(gchop.utils.center(galaxy))
    >>> chopper = gchop.KMeans()
    >>> chopper.decompose(galaxy)

    References
    ----------
    .. [6] Pedregosa et al., Journal of Machine Learning Research 12,
        pp. 2825-2830, 2011.
        `<https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html>`_
    """

    n_components = hparam(default=2)
    init = hparam(default="k-means++")
    n_init = hparam(default=10)
    max_iter = hparam(default=300)
    tol = hparam(default=0.0001)
    verbose = hparam(default=0)
    random_state = hparam(default=None, converter=np.random.default_rng)
    algorithm = hparam(default="lloyd")

    @doc_inherit(GalaxyDecomposerABC.get_attributes)
    def get_attributes(self):
        """
        Notes
        -----
        In this model the parameter space is given by
            normalized_star_energy: normalized specific energy of the stars
            eps: circularity parameter (J_z/J_circ)
            eps_r: projected circularity parameter (J_p/J_circ).
        """
        return ["normalized_star_energy", "eps", "eps_r"]

    @doc_inherit(GalaxyDecomposerABC.split)
    def split(self, X, y, attributes):
        """
        Notes
        -----
        The attributes used by the kmeans model are described in detail in the
        class documentation.
        """
        random_state = np.random.RandomState(self.random_state.bit_generator)

        kmeans = cluster.KMeans(
            n_clusters=self.n_components,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=random_state,
            algorithm=self.algorithm,
        )
        labels = kmeans.fit_predict(X)
        return labels, None
