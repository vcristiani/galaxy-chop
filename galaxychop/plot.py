# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for the galaxy object."""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import numpy as np

import pandas as pd

import seaborn as sns

from . import utils

# =============================================================================
# ACCESSOR
# =============================================================================


@attr.s(frozen=True, order=False)
class GalaxyPlotter:
    """Make plots of DecisionMatrix."""

    _P_KIND_FORBIDEN_METHODS = ("get_df_and_hue",)

    _galaxy = attr.ib()

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
        if (
            plot_kind.startswith("_")
            or plot_kind in self._P_KIND_FORBIDEN_METHODS
        ):
            raise ValueError(f"invalid 'plot_kind' name '{plot_kind}'")
        method = getattr(self, plot_kind, None)
        if not callable(method):
            raise ValueError(f"invalid 'plot_kind' name '{plot_kind}'")
        return method(**kwargs)

    # COMMON PLOTS ============================================================

    def get_df_and_hue(self, ptypes, attributes, labels, lmap):
        attributes = ["x", "y", "z"] if attributes is None else attributes

        hue = None  # by default not hue is selected

        # labels es la columna que se va a usar para "resaltar cosas" (hue)
        # si es un str y no estaba en los atributos lo tengo que sacar
        # del dataframe
        if isinstance(labels, str) and labels not in attributes:
            hue = labels
            attributes = np.concatenate((attributes, [labels]))

        # saco todos los atributos en un df
        df = self._galaxy.to_dataframe(ptypes=ptypes, attributes=attributes)

        # ahora puede ser que los labels sean un np array y hay que agregarlo
        # como columna al dataframe y asignar hue al nombre de esta nueva
        # columna
        if hue is None and hasattr(labels, "__iter__"):
            hue = "Hue"  # Hue no esta en pset por lo tanto sirve
            df.insert(0, hue, labels)  # lo chanto como primer columna

        if lmap is not None:
            df[hue] = df[hue].apply(lambda l: lmap.get(l, l))

        return df, hue

    def pairplot(
        self, ptypes=None, attributes=None, labels="ptype", lmap=None, **kwargs
    ):

        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax

    def dis(self, x, y=None, ptypes=None, labels=None, lmap=None, **kwargs):
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.displot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def scatter(self, x, y, ptypes=None, labels=None, lmap=None, **kwargs):
        attributes = [x, y]
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.scatterplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def hist(self, x, y=None, ptypes=None, labels=None, lmap=None, **kwargs):
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.histplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def kde(self, x, y=None, ptypes=None, labels=None, lmap=None, **kwargs):
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.kdeplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    # CICULARITY ==============================================================

    def circ_hist(self, cbins=(0.05, 0.005), **kwargs):
        circ = utils.jcirc(self._galaxy, *cbins)
        ax = sns.histplot(circ.eps, **kwargs)
        ax.set_xlabel(r"$\epsilon$")
        return ax

    def circ_kde(self, cbins=(0.05, 0.005), **kwargs):
        circ = utils.jcirc(self._galaxy, *cbins)
        ax = sns.kdeplot(circ.eps, **kwargs)
        ax.set_xlabel(r"$\epsilon$")
        return ax

    def circularity_components(
        self, cbins=(0.05, 0.005), labels=None, lmap=None, **kwargs
    ):
        circ = utils.jcirc(self._galaxy, *cbins)

        mask = (
            np.isfinite(circ.normalized_star_energy)
            & np.isfinite(circ.eps)
            & np.isfinite(circ.eps_r)
        )

        columns = {
            "Normalized star energy": circ.normalized_star_energy[mask],
            r"$\epsilon$": circ.eps[mask],
            r"$\epsilon_r$": circ.eps_r[mask],
        }

        hue = None

        if labels is not None:
            hue = "Components"
            columns[hue] = labels[np.isfinite(labels)]

        df = pd.DataFrame(columns)

        if labels is not None and lmap is not None:
            df[hue] = df[hue].apply(lambda l: lmap.get(l, l))

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax
