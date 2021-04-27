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
# SINGLE
# =============================================================================


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
# JThreshold CLASS
# #####################################################


class JThreshold(JHistogram):
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


# =============================================================================
# CRISTIANI
# =============================================================================


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
