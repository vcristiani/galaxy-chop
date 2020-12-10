# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module models."""

# #####################################################
# IMPORTS
# #####################################################

from galaxychop.sklearn_models import GCClusterMixin

import numpy as np

from sklearn.base import TransformerMixin


# #####################################################
# GCAbadi CLASS
# #####################################################


class GCAbadi(GCClusterMixin, TransformerMixin):
    """Galaxy chop Abadi class."""

    def __init__(self, n_bin=100):

        self.n_bin = n_bin

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
        # Building the histogram of the circularity parameter.
        h = np.histogram(X[:, 1], self.n_bin, range=(-1.0, 1.0))[0]
        edges = np.round(
            np.histogram(X[:, 1], self.n_bin, range=(-1.0, 1.0))[1], 2
        )
        a_bin = edges[1] - edges[0]
        center = (
            np.histogram(X[:, 1], self.n_bin, range=(-1.0, 1.0))[1][:-1]
            + a_bin / 2.0
        )
        (cero,) = np.where(edges == 0.0)
        m = cero[0]

        X_ind = np.arange(len(X[:, 1]))

        # Building a dictionary: n={} where the IDs of the particles
        # that satisfy the restrictions given by the mask will be stored.
        # So we can then have control over which particles are selected.
        n = {}

        for i in range(0, self.n_bin - 1):
            (mask,) = np.where(
                (X[:, 1] >= edges[i]) & (X[:, 1] < edges[i + 1])
            )
            n["bin" + "%s" % i] = X_ind[mask]

        (mask,) = np.where(
            (X[:, 1] >= edges[self.n_bin - 1]) & (X[:, 1] <= edges[self.n_bin])
        )
        n["bin" + "%s" % (len(center) - 1)] = X_ind[mask]

        # Selection of the particles that belong to the spheroid according to
        # the circularity parameter.
        np.random.seed(10)
        sph = {}

        for i in range(0, m):
            sph["bin" + "%s" % i] = n["bin" + "%s" % i]

        if len(h) >= 2 * m:
            lim_aux = 0
        else:
            lim_aux = 2 * m - len(h)

        for i in range(lim_aux, m):

            if len(n["bin" + "%s" % i]) >= len(
                n["bin" + "%s" % (2 * m - 1 - i)]
            ):
                sph["bin" + "%s" % (2 * m - 1 - i)] = n[
                    "bin" + "%s" % (2 * m - 1 - i)
                ]
            else:
                sph["bin" + "%s" % (2 * m - 1 - i)] = np.random.choice(
                    n["bin" + "%s" % (2 * m - 1 - i)],
                    len(n["bin" + "%s" % i]),
                    replace=False,
                )

        # The rest of the particles are assigned to the disk.
        dsk = n.copy()

        for i in range(0, m):
            # Bins with only spheroid particles are left empty.
            dsk["bin" + "%s" % i] = []

        x = set()
        y = set()

        if len(h) >= 2 * m:
            lim = m
        else:
            lim = len(h) - m

        for i in range(lim, len(sph)):
            x = set(sph["bin" + "%s" % i])
            y = set(n["bin" + "%s" % i])
            y -= x
            y = np.array(list(y))
            dsk["bin" + "%s" % i] = y

        # The indexes of the particles belonging to the spheroid and the disk
        # are saved.
        esf_ = []
        for i in range(len(sph)):
            esf_ = np.concatenate((esf_, sph["bin" + "%s" % (i)]))
        esf_ = np.int_(esf_)

        disk_ = []
        for i in range(len(dsk)):
            disk_ = np.concatenate((disk_, dsk["bin" + "%s" % (i)]))
        disk_ = np.int_(disk_)

        labels = np.empty(len(X))
        labels[esf_] = 0
        labels[disk_] = 1
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