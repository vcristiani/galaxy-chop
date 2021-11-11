# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
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
    """GalaxyChop KMeans class.

    Implementation of Scikit-learn [6]_ K-means as a method for dynamically
    decomposing galaxies.

    Parameters
    ----------
    columns: default=None
        Physical quantities of stellars particles
        used to decompose galaxies.

    **kwargs: key, value mappings
        Other optional keyword arguments are passed through to
        :py:class:`GalaxyDecomposeMixin`, :py:class:`ClusterMixin` and
        :py:class:`KMeans` classes.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
        Index of the cluster each stellar particles belongs to.

    cluster_centers_:
        Original attribute create by the `k-Means` class into
        `scikit-learn` library.
    inertia_:
        Original attribute create by the `k-Means` class into
        `scikit-learn` library.
    n_iter_ :
        Original attribute create by the `k-Means` class into
        `scikit-learn` library.

    Notes
    -----
    n_clusters: type:int.
        The number of clusters to form. Parameter of :py:class:`KMeans` class.
    More information for `KMeans` class:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Examples
    --------
    Example of implementation of CGKMeans Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> chopper = gchop.KMeans(n_clusters=3)
    >>> chopper.decompose(galaxy)
    >>> chopper.labels_
    array([-1, -1,  2, ...,  1,  2,  1])


    References
    ----------
    .. [6] Pedregosa et al., Journal of Machine Learning Research 12,
        pp. 2825-2830, 2011.
        `<https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html>`_
    """

    n_clusters = hparam(default=2)
    init = hparam(default="k-means++")
    n_init = hparam(default=10)
    max_iter = hparam(default=300)
    tol = hparam(default=0.0001)
    verbose = hparam(default=0)
    random_state = hparam(default=None, converter=np.random.default_rng)
    algorithm = hparam(default="auto")

    def get_attributes(self):
        return ["normalized_star_energy", "eps", "eps_r"]

    @doc_inherit(GalaxyDecomposerABC.split)
    def split(self, X, y, attributes):
        random_state = np.random.RandomState(self.random_state.bit_generator)

        kmeans = cluster.KMeans(
            n_clusters=self.n_clusters,
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
