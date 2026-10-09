# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for the galaxy object."""

# =============================================================================
# IMPORTS
# =============================================================================

from astropy import units as u

import attr

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np

import pandas as pd

import seaborn as sns

from ..constants import PLOT_ORDER, make_component_styles, plot_config

# =============================================================================
# ACCESSOR
# =============================================================================


@attr.s(frozen=True, order=False)
class GalaxyPlotter:
    """Make plots of a Galaxy."""

    _P_KIND_FORBIDEN_METHODS = ("get_df_and_hue", "get_sdyn_df_and_hue")

    # Percentile (and its mirror, 100 - it) used to zoom in on the bulk of
    # the plotted data, so a handful of far-out particles don't stretch the
    # axes and shrink everything else down to a speck.
    _ZOOM_PCT = 1

    # Column with each curve's own circular velocity in rotation_curve():
    # the particle type's here, each component's in DecomposedGalaxyPlotter.
    _VCIRC_ATTRIBUTE = "ptype_circular_velocity"

    _galaxy = attr.ib()

    # INTERNAL ================================================================

    def __call__(self, plot_kind="hist2d", **kwargs):
        """Make plots of the galaxy.

        Parameters
        ----------
        kind : str
            The kind of plot to produce, e.g. 'hist' (default), 'kde' or
            'rotation_curve'.

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

    # COMMON UTILS ============================================================

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

    def _add_units_to_labels(self, ax, x, y=None):
        """
        Append each axis's physical unit, in LaTeX, to its label.

        Looks up ``x``/``y`` as attributes of a ``ParticleSet`` to get
        their unit; columns with no such attribute (e.g. ``ptype``) or no
        unit are left alone.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes whose labels get the unit appended.
        x : str
            Column plotted on the x axis.
        y : str or None
            Column plotted on the y axis, if any.
        """
        gal = self._galaxy
        try:
            x_unit = getattr(gal.stars, x).unit.to_string("latex")
            ax.set_xlabel(f"{ax.get_xlabel()} [{x_unit}]")
        except AttributeError:
            pass
        if y is not None:
            try:
                y_unit = getattr(gal.stars, y).unit.to_string("latex")
                ax.set_ylabel(f"{ax.get_ylabel()} [{y_unit}]")
            except AttributeError:
                pass

    def _drop_legend_title(self, ax):
        """Remove the legend title, if any.

        ``seaborn`` plots built from ``hue=`` default to using the hue
        column's name (e.g. ``"ptype"``) as the legend title, which isn't
        meant to be shown.
        """
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(None)

    # COMMON PLOTS ============================================================

    def get_df_and_hue(
        self, ptypes, attributes, lmap, *, galaxy_circular_velocity=False
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
            Names of ``ParticleSet class`` parameters. Each particle set's
            own, self-contained ``ptype_circular_velocity`` is one of them.
        lmap : dict or callable
            Name assignment to the particle types.
        galaxy_circular_velocity : bool, default value = False
            Whether to add the ``galaxy_circular_velocity`` column (see
            ``Galaxy.galaxy_circular_velocity_``). Most plots don't use it, so
            it isn't computed unless asked for.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of galaxy properties with the particle type in the
            ``ptype`` column (already renamed through ``lmap``).
        style : dict
            Keys ``hue_order``, ``palette``, ``linestyles``, ``alphas`` and
            ``linewidths``, indexed by the display names of the particle
            types present in ``df``.
        """
        attributes = ["x", "y", "z"] if attributes is None else attributes
        attributes = list(dict.fromkeys(list(attributes) + ["ptype"]))

        df = self._galaxy.to_dataframe(
            ptypes=ptypes,
            attributes=attributes,
            galaxy_circular_velocity=galaxy_circular_velocity,
        )

        lmap = self._coerce_lmap(lmap)

        present = set(df["ptype"].unique())
        names = {}
        for ptype in PLOT_ORDER:
            if ptype in present:
                names[ptype] = lmap(ptype)

        df["ptype"] = df["ptype"].map(names).astype("category")

        return df, make_component_styles(names)

    def hist(self, x="x", *, ptypes=None, lmap=None, **kwargs):
        """Draw a univariate histogram of a galaxy property.

        Shortcut for ``hist2d(x, y=None, ...)``.

        Parameters
        ----------
        x : keys of ``ParticleSet class`` parameters.
            Variable that specifies positions on the x axis.
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
        return self.hist2d(x, y=None, ptypes=ptypes, lmap=lmap, **kwargs)

    def hist2d(self, x="x", *, y="z", ptypes=None, lmap=None, **kwargs):
        """Draw a histogram of galaxy properties.

        Plot univariate or bivariate histograms to show distributions of
        the galaxy particles, grouped by particle type.

        Parameters
        ----------
        x, y : keys of ``ParticleSet class`` parameters.
            Variables that specify positions on the x and y axes.
            Default value y = 'z'. Use ``y=None`` for a univariate histogram
            (or call ``hist`` directly).
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
        self._drop_legend_title(ax)
        self._add_units_to_labels(ax, x, y)
        self._zoom_to_data(ax, df, x, y)
        ax.set_box_aspect(1)
        return ax

    def kde(self, x="x", *, ptypes=None, lmap=None, **kwargs):
        """Draw a univariate Kernel Density plot of a galaxy property.

        Shortcut for ``kde2d(x, y=None, ...)``.

        Parameters
        ----------
        x : keys of ``ParticleSet class`` parameters.
            Variable that specifies positions on the x axis.
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
        return self.kde2d(x, y=None, ptypes=ptypes, lmap=lmap, **kwargs)

    def kde2d(self, x="x", *, y="z", ptypes=None, lmap=None, **kwargs):
        """Draw a Kernel Density plot of galaxy properties.

        Plot univariate or bivariate distributions using kernel density
        estimation (KDE), as unfilled contours or curves. Each particle type
        has its own style: stars solid, gas dashed, dark matter dotted.

        Parameters
        ----------
        x, y : keys of ``ParticleSet class`` parameters.
            Variables that specify positions on the x and y axes.
            Default value y = 'z'. Use ``y=None`` for a univariate kde
            (or call ``kde`` directly).
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

        # bivariate kde draws contours (linestyles/linewidths), univariate
        # draws curves (linestyle/linewidth)
        ls_key = "linestyle" if y is None else "linestyles"
        lw_key = "linewidth" if y is None else "linewidths"
        for name in style["hue_order"]:
            group = df[df["ptype"] == name]
            group_kws = {
                "color": style["palette"][name],
                ls_key: style["linestyles"][name],
                lw_key: style["linewidths"][name],
                "alpha": style["alphas"][name],
                "label": name,
            }
            group_kws.update(kwargs)
            sns.kdeplot(data=group, x=x, y=y, ax=ax, **group_kws)

        self._add_units_to_labels(ax, x, y)
        self._zoom_to_data(ax, df, x, y)
        ax.set_box_aspect(1)
        if y is None:
            # univariate: real Line2D curves, label= works out of the box
            ax.legend()
        else:
            # bivariate: contour sets don't register their label with
            # matplotlib's legend, so build proxy handles by hand
            handles = [
                Line2D(
                    [],
                    [],
                    color=style["palette"][name],
                    linestyle=style["linestyles"][name],
                    label=name,
                )
                for name in style["hue_order"]
            ]
            ax.legend(handles=handles)
        self._drop_legend_title(ax)
        return ax

    def rotation_curve(self, *, ptypes=None, galaxy=True, lmap=None, **kwargs):
        """Draw the galaxy's rotation curve (circular velocity vs radius).

        Draws two kinds of curves:

        - The whole galaxy's rotation curve, in solid black: circular
          velocity computed from the mass enclosed by stars, dark matter
          and gas pooled together (see ``Galaxy.galaxy_circular_velocity_``).
        - One curve per particle type, in its usual style (stars solid,
          gas dashed, dark matter dotted), computed from that type's own
          mass alone (see ``ParticleSet.ptype_circular_velocity_``). These show
          each component's own contribution, not the galaxy's real
          dynamics.

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
            in ``seaborn.lineplot``. ``ax`` and the line style of every
            curve can be overridden.

        Returns
        -------
        matplotlib.axes.Axes
        """
        df, style = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=["radius", self._VCIRC_ATTRIBUTE],
            lmap=lmap,
            galaxy_circular_velocity=True,
        )

        ax = kwargs.pop("ax", None)
        ax = plt.gca() if ax is None else ax
        kwargs.setdefault("estimator", None)

        if galaxy:
            # whole galaxy: a single curve pooling every particle type
            whole_galaxy = df.sort_values("radius")
            whole_galaxy_kws = {
                **plot_config.galaxy.get_mplstyle(),
                "label": "galaxy",
            }
            whole_galaxy_kws.update(kwargs)
            sns.lineplot(
                data=whole_galaxy,
                x="radius",
                y="galaxy_circular_velocity",
                ax=ax,
                **whole_galaxy_kws,
            )

        # each particle type on its own, with its usual style
        for name in style["hue_order"]:
            group = df[df["ptype"] == name].sort_values("radius")
            group_kws = {
                "color": style["palette"][name],
                "linestyle": style["linestyles"][name],
                "alpha": style["alphas"][name],
                "linewidth": style["linewidths"][name],
                "label": name,
            }
            group_kws.update(kwargs)
            sns.lineplot(
                data=group,
                x="radius",
                y=self._VCIRC_ATTRIBUTE,
                ax=ax,
                **group_kws,
            )

        kpc = u.kpc.to_string("latex")
        ax.set_xlabel(f"radius [{kpc}]")

        kms = (u.km / u.s).to_string("latex")
        ax.set_ylabel(f"circular velocity [{kms}]")

        ax.set_yscale("log")

        ax.legend()
        self._drop_legend_title(ax)

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
            Keys ``hue_order``, ``palette``, ``linestyles``, ``alphas`` and
            ``linewidths``, indexed by the display name of the stars.
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

        return df, make_component_styles(names)

    def sdyn_hist(
        self, x="normalized_star_energy", *, lmap=None, sdyn_kws=None, **kwargs
    ):
        """Draw a univariate histogram of a stellar dynamics property.

        Shortcut for ``sdyn_hist2d(x, y=None, ...)``.

        Parameters
        ----------
        x : keys of ``GalaxyStellarDynamics`` dataframe.
            Variable that specifies positions on the x axis.
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
        return self.sdyn_hist2d(
            x, y=None, lmap=lmap, sdyn_kws=sdyn_kws, **kwargs
        )

    def sdyn_hist2d(
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
            histogram (or call ``sdyn_hist`` directly).
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
        self._drop_legend_title(ax)
        self._zoom_to_data(ax, df, x, y)
        ax.set_box_aspect(1)
        return ax

    def sdyn_kde(
        self, x="normalized_star_energy", *, lmap=None, sdyn_kws=None, **kwargs
    ):
        """Draw a univariate Kernel Density plot of a stellar dynamics \
        property.

        Shortcut for ``sdyn_kde2d(x, y=None, ...)``.

        Parameters
        ----------
        x : keys of ``GalaxyStellarDynamics`` dataframe.
            Variable that specifies positions on the x axis.
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
        return self.sdyn_kde2d(
            x, y=None, lmap=lmap, sdyn_kws=sdyn_kws, **kwargs
        )

    def sdyn_kde2d(
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
            Default value y = 'eps'. Use ``y=None`` for a univariate kde
            (or call ``sdyn_kde`` directly).
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

        ax = kwargs.pop("ax", None)
        ax = plt.gca() if ax is None else ax
        kwargs.setdefault("fill", False)

        # bivariate kde draws contours (linestyles/linewidths), univariate
        # draws curves (linestyle/linewidth)
        ls_key = "linestyle" if y is None else "linestyles"
        lw_key = "linewidth" if y is None else "linewidths"
        for name in style["hue_order"]:
            group = df[df["ptype"] == name]
            group_kws = {
                "color": style["palette"][name],
                ls_key: style["linestyles"][name],
                lw_key: style["linewidths"][name],
                "alpha": style["alphas"][name],
                "label": name,
            }
            group_kws.update(kwargs)
            sns.kdeplot(data=group, x=x, y=y, ax=ax, **group_kws)

        self._zoom_to_data(ax, df, x, y)
        ax.set_box_aspect(1)
        if y is None:
            # univariate: real Line2D curves, label= works out of the box
            ax.legend()
        else:
            # bivariate: contour sets don't register their label with
            # matplotlib's legend, so build proxy handles by hand
            handles = [
                Line2D(
                    [],
                    [],
                    color=style["palette"][name],
                    linestyle=style["linestyles"][name],
                    label=name,
                )
                for name in style["hue_order"]
            ]
            ax.legend(handles=handles)
        self._drop_legend_title(ax)
        return ax
