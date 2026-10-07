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

import attr

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

# =============================================================================
# ACCESSOR
# =============================================================================


@attr.s(frozen=True, order=False)
class GalaxyPlotter:
    """Make plots of a Galaxy."""

    _P_KIND_FORBIDEN_METHODS = ("get_df_and_hue", "get_sdyn_df_and_hue")

    # Fixed look of the galaxy particle types. Drawing order puts stars last,
    # so they sit on top of the more diffuse components.
    _PTYPE_ORDER = ("dark_matter", "gas", "stars")
    _PTYPE_COLORS = {
        "stars": "tab:red",
        "gas": "tab:blue",
        "dark_matter": "#222222",
    }
    _PTYPE_LINESTYLES = {"stars": "-", "gas": "--", "dark_matter": ":"}

    # Percentile (and its mirror, 100 - it) used to zoom in on the bulk of
    # the plotted data, so a handful of far-out particles don't stretch the
    # axes and shrink everything else down to a speck.
    _ZOOM_PCT = 1

    _galaxy = attr.ib()

    # INTERNAL ================================================================

    def __call__(self, plot_kind="hist", **kwargs):
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

    # Extra room left on each side of the zoomed-in range, as a fraction
    # of that range, so the outermost points don't sit right on the edge.
    _ZOOM_MARGIN = 0.1

    def _zoom_to_data(self, ax, df, x, y=None):
        """
        Zoom the axes in on the bulk of the plotted data.

        Sets the axis limits to the ``[_ZOOM_PCT, 100 - _ZOOM_PCT]``
        percentile range of the plotted columns, instead of letting a
        handful of far-out particles (e.g. tidal debris) stretch the axes
        and shrink the rest of the data down to a speck. A small margin
        (``_ZOOM_MARGIN``) is added on each side so the edge points aren't
        flush against the border.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to zoom.
        df : pandas.DataFrame
            The data actually plotted.
        x : str
            Column plotted on the x axis.
        y : str or None
            Column plotted on the y axis, if any.
        """
        pct = [self._ZOOM_PCT, 100 - self._ZOOM_PCT]
        lo, hi = np.percentile(df[x], pct)
        margin = (hi - lo) * self._ZOOM_MARGIN
        ax.set_xlim(lo - margin, hi + margin)
        if y is not None:
            lo, hi = np.percentile(df[y], pct)
            margin = (hi - lo) * self._ZOOM_MARGIN
            ax.set_ylim(lo - margin, hi + margin)

    def get_df_and_hue(
        self, ptypes, attributes, lmap, *, circular_velocity=False
    ):
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
        circular_velocity : bool, default value = False
            Whether to add the ``circular_velocity`` column (see
            ``Galaxy.circular_velocity_``). Most plots don't use it, so it
            isn't computed unless asked for.

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

        df = self._galaxy.to_dataframe(
            ptypes=ptypes,
            attributes=attributes,
            circular_velocity=circular_velocity,
        )

        lmap = self._coerce_lmap(lmap)

        present = set(df["ptype"].unique())
        names = {}
        for ptype in self._PTYPE_ORDER:
            if ptype in present:
                names[ptype] = lmap(ptype)

        df["ptype"] = df["ptype"].map(names).astype("category")

        return df, self._make_ptype_style(names)

    def _make_ptype_style(self, names):
        """
        Build the fixed plot style for the given particle types.

        Parameters
        ----------
        names : dict
            Maps each particle type to its display name, in drawing order.

        Returns
        -------
        dict
            Keys ``hue_order``, ``palette`` and ``linestyles``, indexed by the
            display names of the particle types.
        """
        return {
            "hue_order": list(names.values()),
            "palette": {names[p]: self._PTYPE_COLORS[p] for p in names},
            "linestyles": {names[p]: self._PTYPE_LINESTYLES[p] for p in names},
        }

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
        self._zoom_to_data(ax, df, x, y)
        ax.set_box_aspect(1)
        return ax

    def kde(self, x="x", *, y="z", ptypes=None, lmap=None, **kwargs):
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
        self._zoom_to_data(ax, df, x, y)
        ax.set_box_aspect(1)
        return ax

    def rotation_curve(self, *, ptypes=None, lmap=None, **kwargs):
        """Draw the galaxy's rotation curve (circular velocity vs radius).

        Circular velocity is computed from the mass enclosed within the
        whole galaxy (stars, dark matter and gas pooled together; see
        ``Galaxy.circular_velocity_``), not from each particle type on
        its own. Each particle type keeps its usual style (stars solid,
        gas dashed, dark matter dotted) to show which radii it covers,
        but all three trace the same underlying curve.

        Parameters
        ----------
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        lmap : dict or callable
            Name assignment to the particle types, e.g.
            ``{"stars": "estrella"}``.
            Default value = None
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.lineplot``. ``ax`` can be overridden.

        Returns
        -------
        matplotlib.axes.Axes
        """
        df, style = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=["radius"],
            lmap=lmap,
            circular_velocity=True,
        )

        ax = kwargs.pop("ax", None)
        ax = plt.gca() if ax is None else ax
        kwargs.setdefault("estimator", None)

        for name in style["hue_order"]:
            group = df[df["ptype"] == name].sort_values("radius")
            group_kws = {
                "color": style["palette"][name],
                "linestyle": style["linestyles"][name],
            }
            group_kws.update(kwargs)
            sns.lineplot(
                data=group,
                x="radius",
                y="circular_velocity",
                ax=ax,
                **group_kws,
            )
        ax.set_xlabel("radius [kpc]")
        ax.set_ylabel("circular velocity [km/s]")
        return ax

    # STELLAR DYNAMICS ========================================================

    def get_sdyn_df_and_hue(self, sdyn_kws, attributes, lmap):
        """
        Dataframe and style constructor for the stellar dynamics plots.

        Only stars have stellar dynamics, so the hue is always the ``stars``
        particle type, with the same fixed look as in the galaxy plots.
        ``lmap`` renames it for display (e.g. ``{"stars": "estrella"}``).

        Parameters
        ----------
        sdyn_kws: dict or None
            Extra parameters for galaxy.stellar_dynamics() method.
        attributes : keys of ``GalaxyStellarDynamics`` dataframe.
            Keys of the normalized specific energy, the circularity parameter
            (J_z/J_circ) and/or the projected circularity parameter
            (J_p/J_circ) of the stellar particles.
        lmap : dict or callable
            Name assignment to the ``stars`` particle type.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of the requested stellar dynamics attributes of the
            stellar particles with finite values, with the particle type in
            the ``ptype`` column (already renamed through ``lmap``).
        style : dict
            Keys ``hue_order``, ``palette`` and ``linestyles``, indexed by the
            display name of the stars.
        """
        sdyn_kws = {} if sdyn_kws is None else sdyn_kws
        sdyn = self._galaxy.stellar_dynamics(**sdyn_kws)
        mask = sdyn.isfinite()

        sdyn_dict = sdyn.to_dict()
        attributes = (
            list(sdyn_dict.keys()) if attributes is None else attributes
        )

        columns = {aname: sdyn_dict[aname][mask] for aname in attributes}
        df = pd.DataFrame(columns)

        names = {"stars": self._coerce_lmap(lmap)("stars")}
        df["ptype"] = pd.Categorical(
            np.full(len(df), names["stars"], dtype=object)
        )

        return df, self._make_ptype_style(names)

    def sdyn_pairplot(
        self,
        *,
        attributes=None,
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

        Parameters
        ----------
        attributes : keys of ``GalaxyStellarDynamics`` dataframe.
            Names of ``GalaxyStellarDynamics`` attributes.
            Default value = None
        lmap : dict or callable
            Name assignment to the stars, e.g. ``{"stars": "estrella"}``.
            Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs :
            Additional keyword arguments are passed and are documented in
            ``seaborn.pairplot``.

        Returns
        -------
        seaborn.axisgrid.PairGrid
        """
        df, style = self.get_sdyn_df_and_hue(
            sdyn_kws=sdyn_kws, attributes=attributes, lmap=lmap
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

    def sdyn_hist(
        self,
        x="normalized_star_energy",
        *,
        y="eps",
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """Draw a histogram of stellar dynamics.

        Plot univariate or bivariate histograms to show distributions of
        the stellar dynamics of the stars.

        Parameters
        ----------
        x, y : keys of ``GalaxyStellarDynamics`` dataframe.
            Variables that specify positions on the x and y axes.
            Default value y = 'eps'. Use ``y=None`` for a univariate
            histogram.
        lmap : dict or callable
            Name assignment to the stars, e.g. ``{"stars": "estrella"}``.
            Default value = None
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
        df, style = self.get_sdyn_df_and_hue(
            sdyn_kws=sdyn_kws, attributes=attributes, lmap=lmap
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
        self._zoom_to_data(ax, df, x, y)
        ax.set_box_aspect(1)
        return ax

    def sdyn_kde(
        self,
        x="normalized_star_energy",
        *,
        y="eps",
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """Draw a Kernel Density plot of stellar dynamics.

        Plot univariate or bivariate distributions of the normalized specific
        energy, the circularity parameter (J_z/J_circ) and/or the projected
        circularity parameter (J_p/J_circ) of the stellar particles using
        kernel density estimation (KDE), as unfilled contours or curves with
        the stars style.

        Parameters
        ----------
        x, y : keys of ``GalaxyStellarDynamics`` dataframe.
            Variables that specify positions on the x and y axes.
            Default value y = 'eps'. Use ``y=None`` for a univariate kde.
        lmap : dict or callable
            Name assignment to the stars, e.g. ``{"stars": "estrella"}``.
            Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.kdeplot``. ``fill`` and the color and line style
            can be overridden.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, style = self.get_sdyn_df_and_hue(
            sdyn_kws=sdyn_kws, attributes=attributes, lmap=lmap
        )

        (name,) = style["hue_order"]
        ls_key = "linestyle" if y is None else "linestyles"
        kde_kws = {
            "fill": False,
            "color": style["palette"][name],
            ls_key: style["linestyles"][name],
        }
        kde_kws.update(kwargs)

        ax = sns.kdeplot(data=df, x=x, y=y, **kde_kws)
        self._zoom_to_data(ax, df, x, y)
        ax.set_box_aspect(1)
        return ax
