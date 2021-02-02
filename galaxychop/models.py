# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module models."""

__all__ = [
    "GCDecomposeMixin",
    "GCClusterMixin",
    "GCAbadi",
    "GCChop",
    "GCKmeans",
    "GCgmm",
    "GCAutogmm",
]

# #####################################################
# IMPORTS
# #####################################################

import numpy as np

from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from . import core


# #####################################################
# GCDecomposeMixin CLASS
# #####################################################


class GCDecomposeMixin:
    """Galaxy chop decompose mixin class."""

    def get_clean_mask(self, X):
        """Clean the Nan value of the circular parameter array."""
        eps = X[:, core.Columns.eps.value]
        (clean,) = np.where(~np.isnan(eps))
        return clean

    def add_dirty(self, X, labels, clean_mask):
        """Complete the labels."""
        eps = X[:, core.Columns.eps.value]
        complete = -np.ones(len(eps), dtype=int)
        complete[clean_mask] = labels
        return complete

    def get_columns(self):
        """Obtain the columns of the quantities to be used."""
        return [
            core.Columns.normalized_energy.value,
            core.Columns.eps.value,
            core.Columns.eps_r.value,
        ]

    def decompose(self, galaxy):
        """Decompose method.

        Validation of the input galaxy instance.

        Parameters
        ----------
        galaxy : `galaxy object`
        """
        if not isinstance(galaxy, core.Galaxy):
            found = type(galaxy)
            raise TypeError(
                f"'galaxy' must be a core.Galaxy instance. Found {found}"
            )

        # retrieve te galaxy as an array os star particles
        X, y = galaxy.values()

        # calculate only the valid values to operate the clustering
        clean_mask = self.get_clean_mask(X)
        X_clean, y_clean = X[clean_mask], y[clean_mask]

        # select only the needed columns
        columns = self.get_columns()
        X_ready = X_clean[:, columns]

        # execute the cluster with the quantities of interest in dynamic
        # decomposition
        self.fit_transform(X_ready, y_clean)

        # retrieve and fix the labels
        labels = self.labels_
        self.labels_ = self.add_dirty(X, labels, clean_mask)

        # return the instance
        return self


# #####################################################
# GCClusterMixin CLASS
# #####################################################


class GCClusterMixin(GCDecomposeMixin, ClusterMixin):
    """Galaxy chop cluster mixin class."""

    pass


# #####################################################
# GCAbadi CLASS
# #####################################################


class GCAbadi(GCClusterMixin, TransformerMixin):
    """Galaxy chop Abadi class."""

    def __init__(self, n_bin=100, digits=2, seed=None):
        """Init function."""
        self.n_bin = n_bin
        self.digits = digits
        self.seed = seed
        self._random = np.random.default_rng(seed=seed)

    def _make_histogram(self, X, n_bin):
        """Build the histrogram of the circularity parameter."""
        eps = X[:, 1]

        full_histogram = np.histogram(eps, n_bin, range=(-1.0, 1.0))

        hist = full_histogram[0]
        edges = np.round(full_histogram[1], self.digits)
        bin_width = edges[1] - edges[0]
        center = full_histogram[1][:-1] + bin_width / 2.0
        bin0 = np.where(edges == 0.0)[0][0]

        return eps, edges, center, bin0, hist

    def _make_count_sph(self, bin0, bin_to_particle):
        """Build the counter-rotating part of the spheroid."""
        sph = {}
        for i in range(bin0):
            sph[i] = bin_to_particle[i]
        return sph

    def _make_corot_sph(self, n_bin, bin0, bin_to_particle, sph):
        """Build the corotating part of the spheroid."""
        lim_aux = 0 if (n_bin >= 2 * bin0) else (2 * bin0 - self.n_bin)

        for count_bin in range(lim_aux, bin0):
            corot_bin = 2 * bin0 - 1 - count_bin

            if len(bin_to_particle[count_bin]) >= len(
                bin_to_particle[corot_bin]
            ):
                sph[corot_bin] = bin_to_particle[corot_bin]
            else:
                sph[corot_bin] = self._random.choice(
                    bin_to_particle[corot_bin],
                    len(bin_to_particle[count_bin]),
                    replace=False,
                )

    def _make_disk(self, bin_to_particle, bin0, n_bin, sph):
        """Build the disk."""
        dsk = bin_to_particle.copy()
        for i in range(bin0):
            dsk[i] = []  # Bins with only spheroid particles are left empty.

        lim = bin0 if (n_bin >= 2 * bin0) else (n_bin - bin0)

        for key in range(lim, len(sph)):
            arr, flt = bin_to_particle[key], sph[key]
            dsk[key] = np.unique(arr[~np.in1d(arr, flt)])
        return dsk

    def fit(self, X, y=None, sample_weight=None):
        """Compute Abadi clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
            Fitted estimator.
        """
        n_bin = self.n_bin

        # Building the histogram of the circularity parameter.
        eps, edges, center, bin0, hist = self._make_histogram(X, n_bin)
        X_ind = np.arange(len(eps))

        # Building a dictionary: n={} where the IDs of the particles
        # that satisfy the restrictions given by the mask will be stored.
        # So we can then have control over which particles are selected.
        bin_to_particle = {}

        for i in range(n_bin - 1):
            (mask,) = np.where((eps >= edges[i]) & (eps < edges[i + 1]))
            bin_to_particle[i] = X_ind[mask]

        (mask,) = np.where(
            (eps >= edges[n_bin - 1]) & (eps <= edges[self.n_bin])
        )
        bin_to_particle[len(center) - 1] = X_ind[mask]

        # Selection of the particles that belong to the spheroid according to
        # the circularity parameter.
        sph = self._make_count_sph(bin0, bin_to_particle)
        self._make_corot_sph(n_bin, bin0, bin_to_particle, sph)

        # The rest of the particles are assigned to the disk.
        dsk = self._make_disk(bin_to_particle, bin0, n_bin, sph)

        # The indexes of the particles belonging to the spheroid and the disk
        # are saved.
        esf = []
        for i in range(len(sph)):
            esf = np.concatenate((esf, sph[i]))
        esf_idx = esf.astype(int)

        disk = []
        for i in range(len(dsk)):
            disk = np.concatenate((disk, dsk[i]))
        disk_idx = disk.astype(int)

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
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def transform(self, X, y=None):
        """Transform method.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed.
        """
        return self


# #####################################################
# GCChop CLASS
# #####################################################


class GCChop(GCAbadi):
    """Galaxy chop Chop class.

    Parameters
    ----------
    eps_cut : float, default=0.6
        Cut-off value in the circularity parameter. Particles with
        eps>eps_cut are assigned to the disk and particles with eps<=eps_cut
        to the spheroid.
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
        """Compute Chop clustering."""
        eps_cut = self.eps_cut

        (esf_idx,) = np.where(X[:, 1] <= eps_cut)
        (disk_idx,) = np.where(X[:, 1] > eps_cut)

        labels = np.empty(len(X), dtype=int)
        labels[esf_idx] = 0
        labels[disk_idx] = 1

        self.labels_ = labels

        return self


# =============================================================================
# SCIKIT_LEARN WRAPPERS
# =============================================================================


class GCKmeans(GCClusterMixin, KMeans):
    """Galaxy chop KMean class."""

    def __init__(self, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns

    def get_columns(self):
        """Obtain the columns of the quantities to be used."""
        if self.columns is None:
            return super().get_columns()
        return self.columns


class GCgmm(GCDecomposeMixin, GaussianMixture):
    """Galaxy chop Gaussian Mixture Model class."""

    def __init__(self, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns

    def get_columns(self):
        """Obtain the columns of the quantities to be used."""
        if self.columns is None:
            return super().get_columns()
        return self.columns

    def fit_transform(self, X, y=None):
        """Transform method."""
        labels = self.fit(X).predict(X)
        self.labels_ = labels
        return self


class GCAutogmm(GCgmm):
    """Galaxy chop auto-gmm class."""

    def __init__(self, c_bic=0.1, **kwargs):
        super().__init__(**kwargs)
        self.c_bic = c_bic
        self.n_components = _get_number_of_gaussians()


    def _get_number_of_gaussians(self, X):
        "Automatic selection of the number of gaussians."
        componenst_to_try = np.arange(2, 16)
        bic_med = np.empty(len(componenst_to_try))

        for i in componenst_to_try:
            # Implementation of gmm for all possible components of the method.
            gmm = GaussianMixture(n_components=i, n_init=10)
            gmm.fit(X)
            bic_med[i] = gmm.bic(X) / len(X)

        bic_min = np.sum(bic_med[-5:]) / 5.0
        delta_bic = bic_med - bic_min

        # Criteria for the choice of the number of gaussians.
        c_bic = self.c_bic
        mask = np.where(delta_bic <= c_bic)[0]

        # nro de componentes
        number_of_gaussians = np.min(componenst_to_try[mask])

        return number_of_gaussians


    def fit_transform(self, X, y=None):
        """Transform method."""
        gmm = GaussianMixture(n_components=self.n_components, n_init=10)
        labels = gmm.fit(X).predict(X)
        self.labels_ = labels
        return self
