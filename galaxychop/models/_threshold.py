# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Monodimensional Models."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin

from ._base import GalaxyDecomposeMixin


# =============================================================================
# Circularity Threshold
# =============================================================================


class JThreshold(GalaxyDecomposeMixin, ClusterMixin, TransformerMixin):
    """GalaxyChop Chop class.

    Implementation of galaxy dynamical decomposition model using
    only the circularity parameter. Tissera et al.(2012) [2]_,
    Marinacci et al.(2014) [3]_, Vogelsberger et al.(2014) [4]_,
    Park et al.(2019) [5]_ .

    Parameters
    ----------
    eps_cut : float, default=0.6
        Cut-off value in the circularity parameter. Particles with
        eps>eps_cut are assigned to the disk and particles with eps<=eps_cut
        to the spheroid.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
        Index of the cluster each stellar particles belongs to.
        Index=0: correspond to galaxy spheroid.
        Index=1: correspond to galaxy disk.

    Examples
    --------
    Example of implementation of Chop Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> gcchop = gchop.JThreshold(eps_cut=0.6)
    >>> gcchop.decompose(galaxy)
    >>> gcchop.labels_
    array([-1, -1,  0, ...,  0,  0,  1])

    References
    ----------
    .. [2] Tissera, P. B., White, S. D. M., and Scannapieco, C.,
        “Chemical signatures of formation processes in the stellar
        populations of simulated galaxies”,
        Monthly Notices of the Royal Astronomical Society, vol. 420, no. 1,
        pp. 255-270, 2012. doi:10.1111/j.1365-2966.2011.20028.x.
        `<https://ui.adsabs.harvard.edu/abs/2012MNRAS.420..255T/abstract>`_
    .. [3] Marinacci, F., Pakmor, R., and Springel, V.,
        “The formation of disc galaxies in high-resolution moving-mesh
        cosmological simulations”, Monthly Notices of the Royal Astronomical
        Society, vol. 437, no. 2, pp. 1750-1775, 2014.
        doi:10.1093/mnras/stt2003.
        `<https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1750M/abstract>`_
    .. [4] Vogelsberger, M., “Introducing the Illustris Project: simulating
        the coevolution of dark and visible matter in the Universe”,
        Monthly Notices of the Royal Astronomical Society, vol. 444, no. 2,
        pp. 1518-1547, 2014. doi:10.1093/mnras/stu1536.
        `<https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V/abstract>`_
    .. [5] Park, M.-J., “New Horizon: On the Origin of the Stellar Disk and
        Spheroid of Field Galaxies at z = 0.7”, The Astrophysical Journal,
        vol. 883, no. 1, 2019. doi:10.3847/1538-4357/ab3afe.
        `<https://ui.adsabs.harvard.edu/abs/2019ApJ...883...25P/abstract>`_
    """

    def __init__(self, eps_cut=0.6):
        """Init function."""
        self.eps_cut = eps_cut
        if self.eps_cut > 1.0 or self.eps_cut < -1.0:
            raise ValueError(
                "The cut-off value in the circularity parameter is not between"
                "(-1,1)."
                "Got eps_cut %d" % (self.eps_cut)
            )

    def fit(self, X, y=None):
        """Compute Chop clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        eps_cut = self.eps_cut

        (esf_idx,) = np.where(X[:, 1] <= eps_cut)
        (disk_idx,) = np.where(X[:, 1] > eps_cut)

        labels = np.empty(len(X), dtype=int)
        labels[esf_idx] = 0
        labels[disk_idx] = 1

        self.labels_ = labels

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def transform(self, X, y=None):
        """Transform method.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed.
        """
        return self
