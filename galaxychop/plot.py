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

import seaborn as sns

# =============================================================================
# ACCESSOR
# =============================================================================


class GalaxyPlotter:
    """Make plots of DecisionMatrix."""

    def __init__(self, galaxy):
        self._galaxy = galaxy

    # INTERNAL ================================================================

    def __call__(self, kind="pairplot", **kwargs):
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
        if kind.startswith("_"):
            raise ValueError(f"invalid kind name '{kind}'")
        method = getattr(self, kind, None)
        if not inspect.ismethod(method):
            raise ValueError(f"invalid kind name '{kind}'")
        return method(**kwargs)

    def pairplot(
        self, attributes=None, hue="ptype", labels=None, ax=None, **kwargs
    ):

        attributes = ["x", "y", "z"] if attributes is None else attributes
        if isinstance(hue, str) and hue not in attributes:
            attributes = attributes + [hue]

        df = self.to_dataframe(attributes)

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")
        kwargs.setdefault("corner", True)

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax
