#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for the galaxy object."""

# =============================================================================
# IMPORTS
# =============================================================================

import inspect

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

# =============================================================================
# ACCESSOR
# =============================================================================


class GalaxyPlotter:
    """Make plots of DecisionMatrix."""

    def __init__(self, galaxy):
        self._galaxy = galaxy

    # INTERNAL ================================================================

    def __call__(self, plot_kind="pairplot", **kwargs):
        """Make plots of the galaxy

        Parameters
        ----------
        kind : str
            The kind of plot to produce:
                - 'pairplot' : pairplot matrix of any coordinates (default)

        **kwargs
            Options to pass to subjacent plotting method.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them
           The ax used by the plot

        """
        if plot_kind.startswith("_"):
            raise ValueError(f"invalid kind name '{kind}'")
        method = getattr(self, plot_kind, None)
        if not inspect.ismethod(method):
            raise ValueError(f"invalid kind name '{kind}'")
        return method(**kwargs)

    # INTERNALS ===============================================================

    def _get_df_and_hue(self, ptypes, attributes, labels):
        attributes = ["x", "y", "z"] if attributes is None else attributes

        hue = None # by default not hue is selected

        # labels es la columna que se va a usar para "resaltar cosas" (hue)
        # si es un str y no estaba en los atributos lo tengo que sacar
        # del dataframe
        if isinstance(labels, str) and labels not in attributes:
            hue = labels
            attributes = np.concatenate((attributes, [labels]))

        # saco todos los atributos en un df
        df = self._galaxy.to_dataframe(ptypes=ptypes, columns=attributes)

        # ahora puede ser los labels sean un np array y hay que agregarlo
        # como columna al dataframe y asignar hue al nombre de esta nueva
        # columna
        if isinstance(labels, (list, np.ndarray)):
            hue = "__labels__"  # labels no esta en pset por lo tanto sirve
            df.insert(0, hue, labels)  # lo chanto como primer columna

        return df, hue

    # PLOTS==== ===============================================================

    def pairplot(
        self, ptypes=None, attributes=None, labels="ptype", ax=None, **kwargs
    ):

        df, hue = self._get_df_and_hue(ptypes, attributes, labels)

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")
        kwargs.setdefault("corner", True)

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax
