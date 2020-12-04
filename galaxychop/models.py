# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module models."""

# #####################################################
# IMPORTS
# #####################################################

from galaxychop import core

import numpy as np

from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans

# #####################################################
# GCDecomposeMixin CLASS
# #####################################################


class GCDecomposeMixin:
    """Galaxy chop decompose mixin class."""

    def decompose(self, galaxy):
        """Decompose method."""
        if not isinstance(galaxy, core.Galaxy):
            found = type(galaxy)
            raise TypeError(
                f"'galaxy' must be a core.Galaxy instance. Found {found}"
            )

        X, y = galaxy.values()
        return self.fit_transform(X, y)


# #####################################################
# GCClusterMixin CLASS
# #####################################################


class GCClusterMixin(GCDecomposeMixin, ClusterMixin):
    """Galaxy chop cluster mixin class."""

    pass


# #####################################################
# GCKmeans CLASS
# #####################################################


class GCKmeans(GCClusterMixin, KMeans):
    """Galaxy chop KMean class."""

    pass


# #####################################################
# GCAbadi CLASS
# #####################################################


class GCAbadi(GCClusterMixin):
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
        # Construimos el histograma del parametro de circularidad.
        h = np.histogram(X, self.n_bin, range=(-1.0, 1.0))[0]
        edges = np.round(np.histogram(X, self.n_bin, range=(-1.0, 1.0))[1], 2)
        a_bin = edges[1] - edges[0]
        center = (
            np.histogram(X, self.n_bin, range=(-1.0, 1.0))[1][:-1]
            + a_bin / 2.0
        )
        (cero,) = np.where(edges == 0.0)
        m = cero[0]

        X_ind = np.arange(len(X))

        # Creamos un diccionario: n={} donde vamos a guardar
        # los ID de las particulas que cumplan
        # con las restricciones que le ponemos a la máscara.

        # Así luego podemos tener control
        # sobre cuales son las partículas que se seleccionan.

        n = {}

        for i in range(0, self.n_bin - 1):
            (mask,) = np.where((X >= edges[i]) & (X < edges[i + 1]))
            n["bin" + "%s" % i] = X_ind[mask]

        (mask,) = np.where(
            (X >= edges[self.n_bin - 1]) & (X <= edges[self.n_bin])
        )
        n["bin" + "%s" % (len(center) - 1)] = X_ind[mask]

        # Seleccionamos las particulas que pertenecen al esferoide en
        # función del parámetro del circularidad y E.
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

        # Al resto de las particulas las asignamos al disco.
        dsk = n.copy()

        for i in range(0, m):
            # Dejamos vacios los bines que sólo tienen particulas
            # del ESFEROIDE.
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

        # Guardamos los indices de las particulas
        # que pertenecen al esferoide y al disco,
        # que estan en el archivo cirularidad_y_L_ID_str(ID).dat.
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
