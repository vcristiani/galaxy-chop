# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


"""Module models."""

# =============================================================================
# IMPORTS
# =============================================================================

from sklearn import cluster

from ._base import GalaxyDecomposeMixin

# =============================================================================
# KNN
# =============================================================================


class KMeans(GalaxyDecomposeMixin, cluster.KMeans):
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

    def __init__(self, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns

    def get_columns(self):
        """Obtain the columns of the quantities to be used.

        Returns
        -------
        columns: list
            Only the needed columns used to decompose galaxies.
        """
        if self.columns is None:
            return super().get_columns()
        return self.columns
