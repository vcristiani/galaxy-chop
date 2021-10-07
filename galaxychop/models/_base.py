# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Common functionalities for galaxy decomposition."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .. import data

# =============================================================================
# BASE MIXIN
# =============================================================================


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
        eps = X[:, data.Columns.eps.value]
        (clean,) = np.where(~np.isnan(eps))
        return clean

    def label_dirty(self, X, labels, clean_mask):
        """Complete the labels of all stellar particles.

        Parameters
        ----------
        X : `np.ndarray(n, 10)`
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
            Particles which not fullfil E <= 0 or -1 < eps < 1 have index = -1.
        """
        eps = X[:, data.Columns.eps.value]
        complete = -np.ones(len(eps), dtype=int)
        complete[clean_mask] = labels
        return complete

    def prepare_values(self, galaxy):
        paramcirc = galaxy.paramcirc

        stars = galaxy.stars.as_array()
        stars = np.delete(stars, np.s_[7:], axis=1)

        n = len(stars)

        X = np.hstack(
            (
                stars,
                paramcirc[0].reshape(
                    n, 1
                ),  # esto es los 3 primeros del jscirc
                paramcirc[1].reshape(n, 1),
                paramcirc[2].reshape(n, 1),
            )
        )
        y = np.zeros(n, dtype=int)
        return X, y

    def get_columns(self):
        """Obtain the columns of the quantities to be used.

        Returns
        -------
        columns: list
            Only the needed columns used to decompose galaxies.

        """
        # m = 0
        # """Masses"""
        # x = 1
        # """x-position"""
        # y = 2
        # """y-position"""
        # z = 3
        # """z-position"""
        # vx = 4
        # """x-component of velocity"""
        # vy = 5
        # """y-component of velocity"""
        # vz = 6
        # """z-component of velocity"""
        # normalized_energy = 7
        # """Normalized specific energy of stars"""
        # eps = 8
        # """Circularity param"""
        # eps_r = 9
        # """Circularity param r"""

        return [
            data.Columns.normalized_energy.value,
            data.Columns.eps.value,
            data.Columns.eps_r.value,
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
        if not isinstance(galaxy, data.Galaxy):
            found = type(galaxy)
            raise TypeError(
                f"'galaxy' must be a data.Galaxy instance. Found {found}"
            )

        # retrieve te galaxy as an array os star particles
        X, y = self.prepare_values(galaxy)

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
        self.labels_ = self.label_dirty(X, labels, clean_mask)

        # return the instance
        return self


import attr


@attr.s
class GalaxyDecomposer:

    bins = attr.ib(default=(0.05, 0.005))

    def decompose(self, galaxy):
        """Decompose method.

        Assign the component of the galaxy to which each particle belongs.
        Validation of the input galaxy instance.

        Parameters
        ----------
        galaxy :
            `galaxy object`
        """

        # retrieve te galaxy as an array os star particles
        X, y = self.prepare_values(galaxy)

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
        self.labels_ = self.label_dirty(X, labels, clean_mask)

        # return the instance
        return self
