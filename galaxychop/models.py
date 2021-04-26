# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module models."""

__all__ = [
    "GalaxyDecomposeMixin",
    "JHistogram",
    "GCChop",
    "JEHistogram",
    "KMeans",
    "GCGmm",
    "GCAutogmm",
]

# #####################################################
# IMPORTS
# #####################################################

import numpy as np

from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn import cluster, mixture

from . import core


# #####################################################
# GalaxyDecomposeMixin CLASS
# #####################################################


class GalaxyDecomposeMixin:
    """GalaxyChop decompose mixin class.

    Implementation of the particle decomposition as a
    method of the class.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of stellar particles.
        Index of the cluster each stellar particles belongs to.
    """

    def get_clean_mask(self, X):
        """Clean the Nan value of the circular parameter array.

        Parameters
        ----------
        X : `np.ndarray(n,10)`
            2D array where each file it is a different stellar particle and
            each column is a parameter of the particles:
            (m_s, x_s, y_s, z_s, vx_s, vy_s, vz_z, E_s, eps_s, eps_r_s)

        Return
        ------
        clean: np.ndarray(n_m: number of particles with E<=0 and -1<eps<1).
            Mask: Index of valid stellar particles to operate the clustering.
        """
        eps = X[:, core.Columns.eps.value]
        (clean,) = np.where(~np.isnan(eps))
        return clean

    def add_dirty(self, X, labels, clean_mask):
        """Complete the labels of all stellar particles.

        Parameters
        ----------
        X : `np.ndarray(n,10)`
            2D array where each file it is a diferent stellar particle and
            each column is a parameter of the particles:
            (m_s, x_s, y_s, z_s, vx_s, vy_s, vz_z, E_s, eps_s, eps_r_s)
        labels: `np.ndarray(n)`, n: number of stellar particles.
            Index of the cluster each stellar particles belongs to.
        clean_mask: np.ndarray(n: number of particles with E<=0 and -1<eps<1).
            Mask: Only valid particles to operate the clustering.

        Return
        ------
        complete: np.ndarray(n: number of particles with E<=0 and -1<eps<1).
            Complete index of the cluster each stellar particles belongs to.
            Particles which not fulfil E<=0 or -1<eps<1 have index=-1.
        """
        eps = X[:, core.Columns.eps.value]
        complete = -np.ones(len(eps), dtype=int)
        complete[clean_mask] = labels
        return complete

    def get_columns(self):
        """Obtain the columns of the quantities to be used.

        Returns
        -------
        columns: list
            Only the needed columns used to decompose galaxies.

        """
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
        galaxy :
            `galaxy object`
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
# JHistogram CLASS
# #####################################################


class JHistogram(GalaxyDecomposeMixin, ClusterMixin, TransformerMixin):
    """GalaxyChop Abadi class.

    Implementation of galaxy dynamical decomposition model described in
    Abadi et al. (2003) [1]_.

    Parameters
    ----------
    n_bin: default=100
        Number of bins needed to build the circularity parameter histogram.
    digits: int, default=2
        Number of decimals to which an array is rounded.
    seed: int, default=None
        Seed to initialize the random generator.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
        Index of the cluster each stellar particles belongs to.
        Index=0: correspond to galaxy spheroid.
        Index=1: correspond to galaxy disk.

    Examples
    --------
    Example of implementation of Abadi Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> chopper = gchop.JHistogram(n_bin=100, digits=2, seed=None)
    >>> chopper.decompose(galaxy)
    >>> chopper.labels_
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
        sph = sph.copy()

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
        return sph

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

        # Building a dictionary: bin_to_particle={} where the IDs of the
        # particles that satisfy the restrictions given by the mask will be
        # stored. So we can then have control over which particles are
        # selected.
        bin_to_particle = {}

        for i in range(n_bin - 1):
            (mask,) = np.where((eps >= edges[i]) & (eps < edges[i + 1]))
            bin_to_particle[i] = X_ind[mask]

        # This considers the right edge of the last bin.
        (mask,) = np.where((eps >= edges[n_bin - 1]) & (eps <= edges[n_bin]))
        bin_to_particle[len(center) - 1] = X_ind[mask]

        # Selection of the particles that belong to the spheroid according to
        # the circularity parameter.
        sph = self._make_count_sph(bin0, bin_to_particle)
        sph = self._make_corot_sph(n_bin, bin0, bin_to_particle, sph)

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


# #####################################################
# GCChop CLASS
# #####################################################


class GCChop(JHistogram):
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
    >>> gcchop = gchop.GCChop(eps_cut=0.6)
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


# #####################################################
# JEHistogram CLASS
# #####################################################


class JEHistogram(JHistogram):
    """GalaxyChop Cristiani class.

    Implementation of a modification of Abadi galaxy dynamical
    decomposition model using the circularity parameter and
    specific energy distribution.

    Parameters
    ----------
    n_bin_E: default=20
        Number of bins needed to build the normalised specific
        energy histogram.
    **kwargs: key, value mappings
        Other optional keyword arguments are passed through to
        :py:class:`JHistogram` classes.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
        Index of the cluster each stellar particles belongs to.
        Index=0: correspond to galaxy spheroid.
        Index=1: correspond to galaxy disk.

    Examples
    --------
    Example of implementation of Cristini Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> chopper = gchop.JEHistogram()
    >>> chopper.decompose(galaxy)
    >>> gcristiani.labels_
    array([-1, -1,  0, ...,  0,  0,  1])
    """

    def __init__(self, n_bin_E=20, **kwargs):
        super().__init__(**kwargs)
        self.n_bin_E = n_bin_E

    def _make_corot_sph(
        self,
        n_bin,
        bin0,
        bin_to_particle,
        sph,
        normalized_energy,
        n_bin_E,
        X_ind,
    ):
        """Build the corotating part of the spheroid."""
        # We build the corotating sph
        lim_aux = 0 if (n_bin >= 2 * bin0) else (2 * bin0 - n_bin)

        for count_bin in range(lim_aux, bin0):
            corot_bin = 2 * bin0 - 1 - count_bin
            # If the length of the bin contrarrot >= length of the bin corrot,
            # then we assign all particles in the bin corrot to the sph.
            if len(bin_to_particle[count_bin]) >= len(
                bin_to_particle[corot_bin]
            ):
                sph[corot_bin] = bin_to_particle[corot_bin]
            # Otherwise, we look at the energy distributions of both bins to
            # make the selection.
            else:
                energy_hist_count, energy_edges_count = np.histogram(
                    normalized_energy[np.int_(bin_to_particle[count_bin])],
                    bins=n_bin_E,
                    range=(-1.0, 0.0),
                )
                energy_hist_corr, energy_edges_corr = np.histogram(
                    normalized_energy[np.int_(bin_to_particle[corot_bin])],
                    bins=n_bin_E,
                    range=(-1.0, 0.0),
                )
                # For a fixed contr and corr bin, we go through the energy bins
                # and select particles and add them to the aux0 list.
                aux0 = []
                for j in range(n_bin_E):
                    if energy_hist_count[j] != 0:
                        # If the energy bin length contr >= the energy bin
                        # length contr, we add all particles from the corr
                        # energy bin to the list of particles in the corr bin.
                        if energy_hist_count[j] >= energy_hist_corr[j]:
                            (energy_mask,) = np.where(
                                (
                                    normalized_energy[
                                        np.int_(bin_to_particle[corot_bin])
                                    ]
                                    >= energy_edges_corr[j]
                                )
                                & (
                                    normalized_energy[
                                        np.int_(bin_to_particle[corot_bin])
                                    ]
                                    < energy_edges_corr[j + 1]
                                )
                            )

                            aux1 = X_ind[np.int_(bin_to_particle[corot_bin])][
                                energy_mask
                            ]
                            aux0 = np.concatenate((aux0, aux1), axis=None)
                        # Otherwise we make a random selection in the energy
                        # bin to add to the list.
                        else:
                            (energy_mask,) = np.where(
                                (
                                    normalized_energy[
                                        np.int_(bin_to_particle[corot_bin])
                                    ]
                                    >= energy_edges_corr[j]
                                )
                                & (
                                    normalized_energy[
                                        np.int_(bin_to_particle[corot_bin])
                                    ]
                                    < energy_edges_corr[j + 1]
                                )
                            )

                            aux1 = self._random.choice(
                                X_ind[np.int_(bin_to_particle[corot_bin])][
                                    energy_mask
                                ],
                                energy_hist_count[j],
                                replace=False,
                            )
                            aux0 = np.concatenate((aux0, aux1), axis=None)
                    # If there are no particles in the contr energy bin, we do
                    # not add anything to the list.
                    else:
                        aux1 = []
                        aux0 = np.concatenate((aux0, aux1), axis=None)
                # We assign all the particles in the list, selected by energy,
                # to the bin corr of the sph.
                sph[corot_bin] = aux0

    def fit(self, X, y=None):
        """Compute Cristiani clustering.

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
        n_bin = self.n_bin
        n_bin_E = self.n_bin_E

        normalized_energy = X[:, 0]

        # Building the histogram of the circularity parameter.
        eps, edges, center, bin0, hist = self._make_histogram(X, n_bin)
        X_ind = np.arange(len(eps))

        # Building a dictionary: bin_to_particle={} where the IDs of the
        # particles that satisfy the restrictions given by the mask will be
        # stored. So we can then have control over which particles are
        # selected.
        bin_to_particle = {}

        for i in range(n_bin - 1):
            (mask,) = np.where((eps >= edges[i]) & (eps < edges[i + 1]))
            bin_to_particle[i] = X_ind[mask]

        # This considers the right edge of the last bin.
        (mask,) = np.where((eps >= edges[n_bin - 1]) & (eps <= edges[n_bin]))
        bin_to_particle[len(center) - 1] = X_ind[mask]

        # Selection of the particles that belong to the spheroid according to
        # the circularity parameter and normalized energy.

        # The counter-rotating spheroid is constructed.
        sph = self._make_count_sph(bin0, bin_to_particle)
        # Ther corotating spheroid is constructed
        self._make_corot_sph(
            n_bin,
            bin0,
            bin_to_particle,
            sph,
            normalized_energy,
            n_bin_E,
            X_ind,
        )

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

        # Component labels are assigned
        labels = np.empty(len(X), dtype=int)
        labels[esf_idx] = 0
        labels[disk_idx] = 1

        self.labels_ = labels

        return self


# =============================================================================
# SCIKIT_LEARN WRAPPERS
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


class GCGmm(GalaxyDecomposeMixin, mixture.GaussianMixture):
    """GalaxyChop Gaussian Mixture Model class.

    Implementation of the method for dynamically decomposing galaxies
    described by Obreja et al.(2018) [7]_ .

    Parameters
    ----------
    columns: default=None
        Physical quantities of stellars particles
        used to decompose galaxies.

    **kwargs: key, value mappings
        Other optional keyword arguments are passed through to
        :py:class:`GalaxyDecomposeMixin` and :py:class:`GaussianMixture` classes.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
        Index of the cluster each stellar particles belongs to.
    weights_ :
        Original attribute create by the `GaussianMixture`
        class into `scikit-learn` library.
    means_ :
        Original attribute create by the `GaussianMixture` class into
        `scikit-learn` library.
    covariances_ :
        Original attribute create by the `GaussianMixture`
        class into `scikit-learn` library.
    precisions_ :
        Original attribute create by the `GaussianMixture` class into
        `scikit-learn` library.
    precisions_cholesky_ :
        Original attribute create by the `GaussianMixture`
        class into `scikit-learn` library.
    converged_ :
        Original attribute create by the `GaussianMixture` class into
        `scikit-learn` library.
    n_iter_ :
        Original attribute create by the `GaussianMixture` class into
        `scikit-learn` library.
    lower_bound_ :
        Original attribute create by the `GaussianMixture`
        class into `scikit-learn` library.

    Notes
    -----
    n_components : int, default=1
        The number of mixture components. Parameter of
        :py:class:`GaussianMixture` class.
    More information for `GaussianMixture` class:
        https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

    Examples
    --------
    Example of implementation of CGGmm Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> gcgmm = gchop.GCGmm(n_components=3)
    >>> gcgmm.decompose(galaxy)
    >>> gcgmm.labels_
    array([-1, -1,  2, ...,  1,  2,  1])

    References
    ----------
    .. [7] Obreja, A., “Introducing galactic structure finder: the multiple
        stellar kinematic structures of a simulated Milky Way mass galaxy”,
        Monthly Notices of the Royal Astronomical Society, vol. 477, no. 4,
        pp. 4915–4930, 2018. doi:10.1093/mnras/sty1022.
        `<https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.4915O/abstract>`_
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

    def fit_transform(self, X, y=None):
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
        labels = self.fit(X).predict(X)
        self.labels_ = labels
        return self


class GCAutogmm(GalaxyDecomposeMixin, ClusterMixin, TransformerMixin):
    """GalaxyChop auto-gmm class.

    Implementation of the method for dynamically decomposing galaxies
    described by Du et al.(2019) [8]_ .

    Parameters
    ----------
    c_bic: float, default=0.1
        Cut value of the criteria for the automatic choice of
        the number of gaussians.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
        Index of the cluster each stellar particles belongs to.
        Index=0: correspond to galaxy stellar halo.
        Index=1: correspond to galaxy bulge.
        Index=2: correspond to galaxy cold disk.
        Index=3: correspond to galaxy warm disk.

    probability: np.ndarray(n,4), n:number of particles with E<=0 and -1<eps<1.
        Probability of each stellar particle to belong to each
        component of the galaxy.

    probability_of_gaussianmixturey: `np.ndarray(n_particles, n_gaussians)`.
        Probability of each stellar particle (with E<=0 and -1<eps<1) to belong
        to each gaussian.

    bic_med_: np.ndarray(number of component to try,1).
        BIC parameter(len(X)).
    gausians_: tuple(n_gausians).
        Number of gaussians used to choise the number of clusters.
    bic_min_: float.
        Mean value of BIC(n_c>10).
    delta_bic_: np.ndarray(number of component to try,1)
        `bic_med_` - `bic_min_`.
    mask_:  np.ndarray(number of gaussians).
        Index of components to try that fulfil with c_BIC criteria.
    n_components_: int.
        Number of gaussians automatically selected.
    gcgmm_: object.
        Original `gcgmm_` object create by the `GaussianMixture` class
        with number of cluster automatically selected.

    Examples
    --------
    Example of implementation of CGAutogmm Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> gcautogmm = gchop.GCAutogmm(c_bic=0.1)
    >>> gcautogmm.decompose(galaxy)
    >>> gcautogmm.labels_
    array([-1, -1,  1, ...  0, 0, 3])

    References
    ----------
    .. [8] Du, M., “Identifying Kinematic Structures in Simulated Galaxies
        Using Unsupervised Machine Learning”, The Astrophysical Journal,
        vol. 884, no. 2, 2019. doi:10.3847/1538-4357/ab43cc.
        `<https://ui.adsabs.harvard.edu/abs/2019ApJ...884..129D/abstract>`_
    """

    def __init__(self, c_bic=0.1):
        self.c_bic = c_bic
        self.component_to_try = np.arange(2, 16)
        # self.component_to_try = (
        #     np.arange(2, 16) if component_to_try is None else
        # component_to_try
        # )

    def fit(self, X, y=None):
        """Compute GCAutogmm clustering.

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
        bic_med = np.empty(len(self.component_to_try))
        gausians = []
        for i in self.component_to_try:
            # Implementation of gmm for all possible components of the method.
            gmm = mixture.GaussianMixture(
                n_components=i, n_init=10, random_state=0
            )
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
        gcgmm_ = mixture.GaussianMixture(
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
