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

import seaborn as sns

# =============================================================================
# ACCESSOR
# =============================================================================


@attr.s(frozen=True, slots=True, order=False)
class GalaxyPlotter:
    """Make plots of DecisionMatrix."""

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
        if plot_kind.startswith("_"):
            raise ValueError(f"invalid 'plot_kind' name '{plot_kind}'")
        method = getattr(self, plot_kind, None)
        if not callable(method):
            raise ValueError(f"invalid 'plot_kind' name '{plot_kind}'")
        return method(**kwargs)

    # INTERNALS ===============================================================

    def _get_df_and_hue(self, ptypes, attributes, labels):
        attributes = ["x", "y", "z"] if attributes is None else attributes

        hue = None  # by default not hue is selected

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

    def pairplot(self, ptypes=None, attributes=None, labels="ptype", **kwargs):

        df, hue = self._get_df_and_hue(ptypes, attributes, labels)

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax
