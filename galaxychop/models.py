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
    "GCGmm",
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
    """GalaxyChop decompose mixin class.

    Implementation of the particle decomposition as a
    method of the class.
    """

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

        Assign the component of the galaxy to which each particle belongs.
        Validation of the input galaxy instance.

        Parameters
        ----------
        galaxy : `galaxy object`

        Return
        ------
        labels_: `np.ndarray(n)`, n: number of stellar particles.
        Index of the cluster each stellar particles belongs to.
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
    """GalaxyChop cluster mixin class."""

    pass


# #####################################################
# GCAbadi CLASS
# #####################################################


class GCAbadi(GCClusterMixin, TransformerMixin):
    """GalaxyChop Abadi class.

    Implementation of galaxy dynamical decomposition model described in
    Abadi et al. (2003) [1]_.

    Examples
    --------
    Example of implementation of Abadi Model.

    >>> gal0 = gc.Galaxy(...)
    >>> gcabadi = gc.GCAbadi(n_bin=100, digits=2, seed=None)
    >>> gcabadi.decompose(gal0)
    >>> labels = gcabadi.labels_
    >>> print(labels)
    array([-1, -1,  0, ...,  0,  0,  1])

    References
    ----------
    .. [1] Abadi, M. G., Navarro, J. F., Steinmetz, M., and Eke, V. R.,
        “Simulations of Galaxy Formation in a Λ Cold Dark Matter Universe. II.
        The Fine Structure of Simulated Galactic Disks”, The Astrophysical
        Journal, vol. 597, no. 1, pp. 21–34, 2003. doi:10.1086/378316.
        `<https://ui.adsabs.harvard.edu/abs/2003ApJ...597...21A/abstract>`_
    """

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
        """Compute Abadi model clustering.

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

    Examples
    --------
    Example of implementation of Chop Model.

    >>> gal0 = gc.Galaxy(...)
    >>> gcchop = gc.GCChop(eps_cut=0.6)
    >>> gcchop.decompose(gal0)
    >>> labels = gcchop.labels_
    >>> print(labels)
    array([-1, -1,  0, ...,  0,  0,  1])

    References
    ----------
    .. [2] Tissera, P. B., White, S. D. M., and Scannapieco, C.,
        “Chemical signatures of formation processes in the stellar
        populations of simulated galaxies”,
        Monthly Notices of the Royal Astronomical Society, vol. 420, no. 1,
        pp. 255–270, 2012. doi:10.1111/j.1365-2966.2011.20028.x.
        `<https://ui.adsabs.harvard.edu/abs/2012MNRAS.420..255T/abstract>`_
    .. [3] Marinacci, F., Pakmor, R., and Springel, V.,
        “The formation of disc galaxies in high-resolution moving-mesh
        cosmological simulations”, Monthly Notices of the Royal Astronomical
        Society, vol. 437, no. 2, pp. 1750–1775, 2014.
        doi:10.1093/mnras/stt2003.
        `<https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1750M/abstract>`_
    .. [4] Vogelsberger, M., “Introducing the Illustris Project: simulating
        the coevolution of dark and visible matter in the Universe”,
        Monthly Notices of the Royal Astronomical Society, vol. 444, no. 2,
        pp. 1518–1547, 2014. doi:10.1093/mnras/stu1536.
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
    """GalaxyChop KMeans class.

    Implementation of Skitlearn [6]_ K-means as a method for dynamically
    decomposing galaxies.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form.

    Examples
    --------
    Example of implementation of CGKMeans Model.

    >>> gal0 = gc.Galaxy(...)
    >>> gckmeans = gc.GCKmeans(n_clusters=3)
    >>> gckmeans.decompose(gal0)
    >>> labels = gckmeans.labels_
    >>> print(labels)
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
        """Obtain the columns of the quantities to be used."""
        if self.columns is None:
            return super().get_columns()
        return self.columns


class GCGmm(GCDecomposeMixin, GaussianMixture):
    """GalaxyChop Gaussian Mixture Model class.

    Implementation of the method for dynamically decomposing galaxies
    described by Obreja et al.(2019) [7]_ .

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    Examples
    --------
    Example of implementation of CGGmm Model.

    >>> gal0 = gc.Galaxy(...)
    >>> gcgmm = gc.GCGmm(n_components=3)
    >>> gcgmm.decompose(gal0)
    >>> labels = gcgmm.labels_
    >>> print(labels)
    array([-1, -1,  2, ...,  1,  2,  1])

    References
    ----------
    .. [7] Obreja, A., “NIHAO XVI: the properties and evolution of
        kinematically selected discs, bulges, and stellar haloes”,
        Monthly Notices of the Royal Astronomical Society, vol. 487,
        no. 3, pp. 4424–4456, 2019. doi:10.1093/mnras/stz1563.
        `<https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.4424O/abstract>`_
    """

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


class GCAutogmm(GCClusterMixin, TransformerMixin):
    """GalaxyChop auto-gmm class.

    Implementation of the method for dynamically decomposing galaxies
    described by Du et al.(2019) [8]_ .

    Examples
    --------
    Example of implementation of CGAutogmm Model.

    >>> gal0 = gc.Galaxy(...)
    >>> gcautogmm = gc.GCAutogmm(c_bic=0.1)
    >>> gcautogmm.decompose(gal0)
    >>> labels = gcautogmm.labels_
    >>> print(labels)
    array([-1, -1,  2, ...,  1,  2,  1])

    References
    ----------
    .. [8] Du, M., “Identifying Kinematic Structures in Simulated Galaxies
        Using Unsupervised Machine Learning”, The Astrophysical Journal,
        vol. 884, no. 2, 2019. doi:10.3847/1538-4357/ab43cc.
        `<https://ui.adsabs.harvard.edu/abs/2019ApJ...884..129D/abstract>`_
    """

    def __init__(self, c_bic=0.1, component_to_try=None):
        self.c_bic = c_bic
        self.component_to_try = (
            np.arange(2, 16) if component_to_try is None else component_to_try
        )

    def fit(self, X, y=None):
        """Compute clustering."""
        bic_med = np.empty(len(self.component_to_try))
        gausians = []
        for i in self.component_to_try:
            # Implementation of gmm for all possible components of the method.
            gmm = GaussianMixture(n_components=i, n_init=10, random_state=0)
            gmm.fit(X)
            bic_med[i - 2] = gmm.bic(X) / len(X)
            gausians.append(gmm)

        bic_min = np.sum(bic_med[-5:]) / 5.0
        delta_bic_ = bic_med - bic_min

        # Criteria for the choice of the number of gaussians.
        c_bic = self.c_bic
        mask = np.where(delta_bic_ <= c_bic)[0]

        # Number of components
        number_of_gaussians = np.min(self.component_to_try[mask])

        # Clustering with gaussian mixture and the parameters obtained.
        gcgmm_ = GaussianMixture(
            n_components=number_of_gaussians,
            random_state=0,
        )
        gcgmm_.fit(X)

        # store all in the instances
        self.bic_med_ = bic_med
        self.gausians_ = tuple(gausians)
        self.bic_min_ = bic_min
        self.delta_bic_ = delta_bic_
        self.c_bic_ = c_bic
        self.mask_ = mask
        self.n_components_ = number_of_gaussians
        self.gcgmm_ = gcgmm_

        return self

    def transform(self, X, y=None):
        """Transform method."""
        n_components = self.n_components_
        center = self.gcgmm_.means_
        predict_proba = self.gcgmm_.predict_proba(X)

        # We add up the probabilities to obtain the classification of the
        # different particles.
        halo = np.zeros(len(X))
        bulge = np.zeros(len(X))
        cold_disk = np.zeros(len(X))
        warm_disk = np.zeros(len(X))

        for i in range(0, n_components):
            if center[i, 1] >= 0.85:
                cold_disk = cold_disk + predict_proba[:, i]
            if (center[i, 1] < 0.85) & (center[i, 1] >= 0.5):
                warm_disk = warm_disk + predict_proba[:, i]
            if (center[i, 1] < 0.5) & (center[i, 0] >= -0.75):
                halo = halo + predict_proba[:, i]
            if (center[i, 1] < 0.5) & (center[i, 0] < -0.75):
                bulge = bulge + predict_proba[:, i]

        probability = np.column_stack((halo, bulge, cold_disk, warm_disk))
        labels = np.empty(len(X), dtype=int)

        for i in range(len(X)):
            labels[i] = probability[i, :].argmax()

        self.probability_of_gaussianmixture = predict_proba
        self.probability = probability
        self.labels_ = labels

        return self
