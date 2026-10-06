# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for the galaxy object."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import OrderedDict

import attr

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from .. import models

# =============================================================================
# ACCESSOR
# =============================================================================


@attr.s(frozen=True, order=False)
class GalaxyPlotter:
    """Make plots of a Galaxy."""

    _P_KIND_FORBIDEN_METHODS = ("get_df_and_hue", "get_circ_df_and_hue")
    _DEFAULT_HUE_COLUMN = "Labels"
    _DEFAULT_HUE_COUNT_COLUMN = "LabelsCnt"

    # Fixed look of the galaxy particle types. Drawing order puts stars last,
    # so they sit on top of the more diffuse components.
    _PTYPE_ORDER = ("dark_matter", "gas", "stars")
    _PTYPE_COLORS = {
        "stars": "black",
        "gas": "#7f7f7f",
        "dark_matter": "#bdbdbd",
    }
    _PTYPE_LINESTYLES = {"stars": "-", "gas": "--", "dark_matter": ":"}

    _galaxy = attr.ib()

    # INTERNAL ================================================================

    def __call__(self, plot_kind="pairplot", **kwargs):
        """Make plots of the galaxy.

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

    def _coerce_lmap(self, lmap):
        """
        Convert ``lmap`` into a callable that maps a label to its name.

        Parameters
        ----------
        lmap : dict, callable or None
            Name assignment to the labels. ``None`` keeps the labels as they
            are, and a dict leaves the labels it doesn't contain unchanged.

        Returns
        -------
        callable
            Function that receives a label and returns its display name.

        Raises
        ------
        TypeError
            If ``lmap`` is not a dict, a callable or None.
        """
        if lmap is None:
            return lambda label: label  # identity
        elif isinstance(lmap, dict):
            return lambda label: lmap.get(label, label)
        elif not callable(lmap):
            raise TypeError("'lmap' must be a dict, callable or None")

        return lmap

    def get_df_and_hue(self, ptypes, attributes, lmap):
        """
        Dataframe and style constructor for the galaxy plot implementations.

        The hue is always the particle type. ``lmap`` renames particle types
        for display (e.g. ``{"stars": "estrella"}``).

        Parameters
        ----------
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        attributes : keys of ``ParticleSet class`` parameters.
            Names of ``ParticleSet class`` parameters.
        lmap : dict or callable
            Name assignment to the particle types.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of galaxy properties with the particle type in the
            ``ptype`` column (already renamed through ``lmap``).
        style : dict
            Keys ``hue_order``, ``palette`` and ``linestyles``, indexed by the
            display names of the particle types present in ``df``.
        """
        attributes = ["x", "y", "z"] if attributes is None else attributes
        attributes = list(dict.fromkeys(list(attributes) + ["ptype"]))

        df = self._galaxy.to_dataframe(ptypes=ptypes, attributes=attributes)

        lmap = self._coerce_lmap(lmap)

        present = set(df["ptype"].unique())
        names = {}
        for ptype in self._PTYPE_ORDER:
            if ptype in present:
                names[ptype] = lmap(ptype)

        df["ptype"] = df["ptype"].map(names).astype("category")

        style = {
            "hue_order": list(names.values()),
            "palette": {names[p]: self._PTYPE_COLORS[p] for p in names},
            "linestyles": {names[p]: self._PTYPE_LINESTYLES[p] for p in names},
        }
        return df, style

    def pairplot(self, *, ptypes=None, attributes=None, lmap=None, **kwargs):
        """
        Draw a pairplot of the galaxy properties.

        By default, this function will create a grid of Axes such that each
        numeric variable in data will by shared across the y-axes across a
        single row and the x-axes across a single column. The diagonal
        plots drow a univariate distribution to show the marginal distribution
        of the data in each column.
        The values are grouped by particle type (stars, gas and dark matter).

        Parameters
        ----------
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        attributes : keys of ``ParticleSet class`` parameters.
            Names of ``ParticleSet class`` parameters. Default value = None
        lmap : dict or callable
            Name assignment to the particle types, e.g.
            ``{"stars": "estrella"}``.
            Default value = None
        **kwargs :
            Additional keyword arguments are passed and are documented in
            ``seaborn.pairplot``.

        Returns
        -------
        seaborn.axisgrid.PairGrid
        """
        df, style = self.get_df_and_hue(
            ptypes=ptypes, attributes=attributes, lmap=lmap
        )

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")
        kwargs.setdefault("diag_kws", {"fill": False})

        ax = sns.pairplot(
            data=df,
            hue="ptype",
            hue_order=style["hue_order"],
            palette=style["palette"],
            **kwargs,
        )
        return ax

    def hist(self, x="x", *, y="z", ptypes=None, lmap=None, **kwargs):
        """Draw a histogram of galaxy properties.

        Plot univariate or bivariate histograms to show distributions of
        the galaxy particles, grouped by particle type.

        Parameters
        ----------
        x, y : keys of ``ParticleSet class`` parameters.
            Variables that specify positions on the x and y axes.
            Default value y = 'z'. Use ``y=None`` for a univariate histogram.
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        lmap : dict or callable
            Name assignment to the particle types, e.g.
            ``{"stars": "estrella"}``.
            Default value = None
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, style = self.get_df_and_hue(
            ptypes=ptypes, attributes=attributes, lmap=lmap
        )
        ax = sns.histplot(
            x=x,
            y=y,
            data=df,
            hue="ptype",
            hue_order=style["hue_order"],
            palette=style["palette"],
            **kwargs,
        )
        return ax

    def kde(self, x, *, y=None, ptypes=None, lmap=None, **kwargs):
        """Draw a Kernel Density plot of galaxy properties.

        Plot univariate or bivariate distributions using kernel density
        estimation (KDE), as unfilled contours or curves. Each particle type
        has its own style: stars solid, gas dashed, dark matter dotted.

        Parameters
        ----------
        x, y : keys of ``ParticleSet class`` parameters.
            Variables that specify positions on the x and y axes.
            Default value y = None.
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        lmap : dict or callable
            Name assignment to the particle types, e.g.
            ``{"stars": "estrella"}``.
            Default value = None
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.kdeplot``. ``ax`` and ``fill`` can be overridden.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, style = self.get_df_and_hue(
            ptypes=ptypes, attributes=attributes, lmap=lmap
        )

        ax = kwargs.pop("ax", None)
        ax = plt.gca() if ax is None else ax
        kwargs.setdefault("fill", False)

        # bivariate kde draws contours (linestyles), univariate draws curves
        ls_key = "linestyle" if y is None else "linestyles"
        for name in style["hue_order"]:
            group = df[df["ptype"] == name]
            group_kws = {
                "color": style["palette"][name],
                ls_key: style["linestyles"][name],
            }
            group_kws.update(kwargs)
            sns.kdeplot(data=group, x=x, y=y, ax=ax, **group_kws)
        return ax

    # CICULARITY ==============================================================

    def get_sdyn_df_and_hue(self, sdyn_kws, attributes, labels, lmap):
        """
        Dataframe and Hue constructor for plot implementations.

        Parameters
        ----------
        sdyn_kws: dict or None
            Extra parameters for galaxy.stellar_dynamics() method.
        attributes : keys of ``GalaxyStellarDynamics`` dataframe.
            Keys of the normalized specific energy, the circularity parameter
            (J_z/J_circ) and/or the projected circularity parameter
            (J_p/J_circ) of the stellar particles.
        labels : keys of ``GalaxyStellarDynamics`` dataframe.
            Variable to map plot aspects to different colors.
        lmap :  dict
            Name assignment to the label.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of the normalized specific energy, the circularity
            parameter (J_z/J_circ) and/or the projected circularity parameter
            (J_p/J_circ) of the stellar particles with labels added.
        hue : keys of ``GalaxyStellarDynamics`` dataframe.
            Labels of stellar particles.
        """
        # if we use the components as laberls we need to extract the labels
        # and the lmap if lmap is None
        if isinstance(
            labels,
            (
                getattr(models, "Components", tuple),
                models.DecomposedParticleSet,
            ),
        ):
            if hasattr(labels, "lmap"):
                lmap = labels.lmap if lmap is None else lmap
            labels = labels.labels

        # first we extract the circularity parameters from the galaxy
        # as a dictionary
        sdyn_kws = {} if sdyn_kws is None else sdyn_kws
        sdyn = self._galaxy.stellar_dynamics(**sdyn_kws)
        mask = sdyn.isfinite()

        sdyn_dict = sdyn.to_dict()

        # determine the correct number of attributes
        attributes = (
            list(sdyn_dict.keys()) if attributes is None else attributes
        )
        hue = None

        # labels: column used to map plot aspects to different colors (hue).
        # if is a str and it was not in the attributes but we can retrieve from
        # circ, we add as an attribute
        if isinstance(labels, str):
            hue = labels
            attributes = np.unique(list(attributes) + [labels])

        columns = OrderedDict()
        for aname in attributes:
            columns[aname] = sdyn_dict[aname][mask]

        df = pd.DataFrame(columns)  # here we create the dataframe

        # At this point if "hue" is still "None" we can assume:
        # is an array simply paste it into the dataframe.
        if hue is None and labels is not None:
            # if the labels are passed to me as an array,
            # I only delete the nans and inf.
            labels = np.asarray(labels)
            labels = labels[mask]
            hue = self._DEFAULT_HUE_COLUMN

            # I place it as the first column
            df.insert(0, hue, labels)

        if hue and lmap is not None:
            df[hue] = df[hue].apply(self._coerce_lmap(lmap))

        # for consitency if we have a hue, we use the natural order
        if hue is not None:
            df[hue] = df[hue].astype("category")

            hue_count = df[hue].value_counts()
            df[self._DEFAULT_HUE_COUNT_COLUMN] = df[hue].map(hue_count)

            df = df.sort_values(
                by=self._DEFAULT_HUE_COUNT_COLUMN, ascending=True
            )

        return df, hue

    def sdyn_pairplot(
        self,
        *,
        attributes=None,
        labels=None,
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """
        Draw a pairplot of stellar dynamics.

        By default, this function will create a grid of Axes such that each
        numeric variable in data will by shared across the y-axes across a
        single row and the x-axes across a single column. The diagonal
        plots drow a univariate distribution to show the marginal distribution
        of the data in each column.
        This function groups the values of stellar particles according to some
        keys of ``JCirc`` tuple.

        Parameters
        ----------
        attributes : keys of ``GalaxyStarsDynamics class`` parameters.
            Names of ``GalaxyStarsDynamics class`` parameters.
            Default value = None
        labels : keys of ``JCirc`` tuple.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label. Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs :
            Additional keyword arguments are passed and are documented in
            ``seaborn.pairplot``.

        Returns
        -------
        seaborn.axisgrid.PairGrid
        """
        df, hue = self.get_sdyn_df_and_hue(
            attributes=attributes,
            labels=labels,
            sdyn_kws=sdyn_kws,
            lmap=lmap,
        )

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax

    def sdyn_hist(
        self,
        x="normalized_star_energy",
        *,
        y="eps",
        labels=None,
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """Draw a histogram of stellar dynamics.

        Plot univariate or bivariate histograms to show distributions of
        datasets. This function groups the values of stellar particles
        according to some keys of ``JCirc`` tuple.

        Parameters
        ----------
        x, y : keys of ``JCirc`` tuple.
            Variables that specify positions on the x and y axes.
            Default value y = 'eps'.
        labels : keys of ``JCirc`` tuple.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label. Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_sdyn_df_and_hue(
            sdyn_kws=sdyn_kws,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.histplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def sdyn_kde(
        self,
        x,
        *,
        y=None,
        labels=None,
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """Draw a Kernel Density plot of stellar dynamics.

        Plot univariate or bivariate distributions using kernel density
        estimation (KDE). This plot represents normalized specific energy, the
        circularity parameter (J_z/J_circ) and/or the projected circularity
        parameter (J_p/J_circ)  of the stellar particles using a continuous
        probability density curve in one or more dimensions.
        This function groups the values of stellar particles according
        to some keys of ``JCirc`` tuple.

        Parameters
        ----------
        x, y : keys of ``JCirc`` tuple.
            Variables that specify positions on the x and y axes.
            Default value y = None.
        labels : keys of ``JCirc`` tuple.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label. Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.kdeplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_sdyn_df_and_hue(
            sdyn_kws=sdyn_kws,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.kdeplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax
