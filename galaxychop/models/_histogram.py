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

from ._base import DynamicStarsDecomposerMixin, GalaxyDecomposerABC, hparam
from ..utils import doc_inherit


# =============================================================================
# SINGLE
# =============================================================================


class JHistogram(DynamicStarsDecomposerMixin, GalaxyDecomposerABC):
    """JHistogram class.

    Implementation of galaxy dynamical decomposition model described in
    Abadi et al. (2003) [1]_.

    Parameters
    ----------
    n_bin: int, default=100
        Number of bins needed to build the circularity parameter histogram.
    digits: int, default=2
        Number of decimals to which an array is rounded.
    seed: int, default=None
        Seed to initialize the random generator.

    Notes
    -----
    Index of the cluster each stellar particles belongs to:
        Index=0: correspond to galaxy spheroid.
        Index=1: correspond to galaxy disk.

    Examples
    --------
    Example of implementation of Abadi Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.read_hdf5(...)
    >>> galaxy = gchop.star_align(gchop.center(galaxy))
    >>> chopper = gchop.JHistogram()
    >>> chopper.decompose(galaxy)

    References
    ----------
    .. [1] Abadi, M. G., Navarro, J. F., Steinmetz, M., and Eke, V. R.,
        “Simulations of Galaxy Formation in a Λ Cold Dark Matter Universe. II.
        The Fine Structure of Simulated Galactic Disks”, The Astrophysical
        Journal, vol. 597, no. 1, pp. 21–34, 2003. doi:10.1086/378316.
        `<https://ui.adsabs.harvard.edu/abs/2003ApJ...597...21A/abstract>`_
    """

    n_bin = hparam(default=100)
    digits = hparam(default=2)
    random_state = hparam(default=None, converter=np.random.default_rng)

    def _make_histogram(self, X, n_bin):
        """Build the histrogram of the circularity parameter."""
        eps = X
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
                sph[corot_bin] = self.random_state.choice(
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

    @doc_inherit(GalaxyDecomposerABC.get_attributes)
    def get_attributes(self):
        """
        Notes
        -----
        In this model the parameter space is given by
            eps: circularity parameter (J_z/J_circ)
        """
        return ["eps"]

    @doc_inherit(GalaxyDecomposerABC.split)
    def split(self, X, y, attributes):
        """
        Notes
        -----
        The attributes used by the Abadi model are described in detail in the
        class documentation.
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
            mask = np.where((eps >= edges[i]) & (eps < edges[i + 1]))[0]
            bin_to_particle[i] = X_ind[mask]

        # This considers the right edge of the last bin.
        mask = np.where((eps >= edges[n_bin - 1]) & (eps <= edges[n_bin]))[0]
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

        return labels, None


# =============================================================================
# CRISTIANI
# =============================================================================
class JEHistogram(JHistogram):
    """JEHistogram class.

    Implementation of a modification of Abadi galaxy dynamical decomposition
    model using the circularity parameter and specific energy distribution.

    Parameters
    ----------
    n_bin_E: int, default=20
        Number of bins needed to build the normalised specific
        energy histogram.
    **kwargs: key, value mappings
        Other optional keyword arguments are passed through to
        :py:class:`JHistogram` classes.

    Notes
    -----
    Index of the cluster each stellar particles belongs to:
        Index=0: correspond to galaxy spheroid.
        Index=1: correspond to galaxy disk.

    Examples
    --------
    Example of the implementation of the modified Abadi model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.read_hdf5(...)
    >>> galaxy = gchop.star_align(gchop.center(galaxy))
    >>> chopper = gchop.JEHistogram()
    >>> chopper.decompose(galaxy)
    """

    n_bin_E = hparam(default=20)

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
                            energy_mask = np.where(
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
                            )[0]

                            aux1 = X_ind[np.int_(bin_to_particle[corot_bin])][
                                energy_mask
                            ]
                            aux0 = np.concatenate((aux0, aux1), axis=None)
                        # Otherwise we make a random selection in the energy
                        # bin to add to the list.
                        else:
                            energy_mask = np.where(
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
                            )[0]

                            aux1 = self.random_state.choice(
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

    @doc_inherit(GalaxyDecomposerABC.get_attributes)
    def get_attributes(self):
        """
        Notes
        -----
        In this model the parameter space is given by
            normalized_star_energy: normalized specific energy of the stars
            eps: circularity parameter (J_z/J_circ).
        """
        return ["normalized_star_energy", "eps"]

    @doc_inherit(GalaxyDecomposerABC.split)
    def split(self, X, y, attributes):
        """
        Notes
        -----
        The attributes used by the modified Abadi model are described in detail
        in the class documentation.
        """
        n_bin = self.n_bin
        n_bin_E = self.n_bin_E
        normalized_energy = X[:, 0]

        # Building the histogram of the circularity parameter.
        eps, edges, center, bin0, hist = self._make_histogram(X[:, 1], n_bin)
        X_ind = np.arange(len(eps))

        # Building a dictionary: bin_to_particle={} where the IDs of the
        # particles that satisfy the restrictions given by the mask will be
        # stored. So we can then have control over which particles are
        # selected.
        bin_to_particle = {}

        for i in range(n_bin - 1):
            mask = np.where((eps >= edges[i]) & (eps < edges[i + 1]))[0]
            bin_to_particle[i] = X_ind[mask]

        # This considers the right edge of the last bin.
        mask = np.where((eps >= edges[n_bin - 1]) & (eps <= edges[n_bin]))[0]
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

        return labels, None
