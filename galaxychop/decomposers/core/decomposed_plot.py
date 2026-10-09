# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for decomposed galaxies."""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd

from ...constants import PLOT_ORDER, make_component_styles, plot_config
from ...core.plot import GalaxyPlotter

# =============================================================================
# FUNCTIONS
# =============================================================================


def _component_key(raw_label):
    """
    Normalize a raw component label into a ``plot_config`` key.

    Parameters
    ----------
    raw_label : object
        A component's ``labels`` value (e.g. ``"Cold Disk"``), or its raw
        numeric ``components`` code stringified, when there is no label.

    Returns
    -------
    str
        The matching ``plot_config`` key, or ``"no_component"`` if
        ``raw_label`` doesn't match any of them.
    """
    key = str(raw_label).lower().replace(" ", "_")
    return key if key in plot_config else "no_component"


# =============================================================================
# ACCESSOR
# =============================================================================


class DecomposedGalaxyPlotter(GalaxyPlotter):
    """Make plots of a DecomposedGalaxy, colored by its components.

    Every plot (``hist2d``, ``kde2d``, ``rotation_curve``, etc.) is
    inherited unchanged from ``GalaxyPlotter``: they only ever work
    through the ``df``/``style`` built by ``get_df_and_hue`` and
    ``get_sdyn_df_and_hue``, so overriding those two is enough to make
    every plot group particles by component instead of particle type.
    """

    # rotation_curve() draws each component's own circular velocity, not
    # its particle type's: that one pools every component of the type
    # together, so every component's curve would just replay it
    _VCIRC_ATTRIBUTE = "component_circular_velocity"

    def get_df_and_hue(
        self, ptypes, attributes, lmap, *, galaxy_circular_velocity=False
    ):
        """Dataframe and style constructor, hued by component.

        Same as ``GalaxyPlotter.get_df_and_hue``, except the hue is each
        particle's component (``labels``, or its raw ``components`` code
        when unlabeled) instead of its particle type.

        Parameters
        ----------
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        attributes : keys of ``ParticleSet class`` parameters.
            Names of ``ParticleSet class`` parameters.
        lmap : dict or callable
            Name assignment to the components.
        galaxy_circular_velocity : bool, default value = False
            Whether to add the ``galaxy_circular_velocity`` column.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of galaxy properties with each particle's component
            in the ``ptype`` column (already renamed through ``lmap``).
        style : dict
            Keys ``hue_order``, ``palette``, ``linestyles``, ``alphas``
            and ``linewidths``, indexed by the display names of the
            components present in ``df``.
        """
        attributes = ["x", "y", "z"] if attributes is None else attributes
        attributes = list(
            dict.fromkeys(list(attributes) + ["labels", "components"])
        )

        df = self._galaxy.to_dataframe(
            ptypes=ptypes,
            attributes=attributes,
            galaxy_circular_velocity=galaxy_circular_velocity,
        )

        raw = df["labels"].where(
            df["labels"].notna(), df["components"].astype(str)
        )
        df["ptype"] = raw.map(_component_key)
        df = df.drop(columns=["labels", "components"])

        lmap = self._coerce_lmap(lmap)

        present = set(df["ptype"].unique())
        names = {}
        for key in PLOT_ORDER:
            if key in present:
                names[key] = lmap(key)

        df["ptype"] = df["ptype"].map(names).astype("category")

        return df, make_component_styles(names)

    def get_sdyn_df_and_hue(self, sdyn_kws, attributes, lmap):
        """Dataframe and style constructor for stellar dynamics, by component.

        Same as ``GalaxyPlotter.get_sdyn_df_and_hue``, except the hue is
        each star's component (``labels``, or its raw ``components`` code
        when unlabeled) instead of always being the ``stars`` type.

        Parameters
        ----------
        sdyn_kws : dict or None
            Extra parameters for galaxy.stellar_dynamics() method.
        attributes : keys of ``GalaxyStellarDynamics`` dataframe.
            Keys of the normalized specific energy, the circularity
            parameter (J_z/J_circ) and/or the projected circularity
            parameter (J_p/J_circ) of the stellar particles.
        lmap : dict or callable
            Name assignment to the components.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of the requested stellar dynamics attributes of the
            stellar particles with finite values, with each star's
            component in the ``ptype`` column (already renamed through
            ``lmap``).
        style : dict
            Keys ``hue_order``, ``palette``, ``linestyles``, ``alphas``
            and ``linewidths``, indexed by the display names of the
            components present in ``df``.
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

        stars = self._galaxy.stars
        labels = pd.Series(stars.labels[mask])
        components = pd.Series(stars.components[mask]).astype(str)
        raw = labels.where(labels.notna(), components)
        df["ptype"] = raw.map(_component_key).to_numpy()

        lmap = self._coerce_lmap(lmap)

        present = set(df["ptype"].unique())
        names = {}
        for key in PLOT_ORDER:
            if key in present:
                names[key] = lmap(key)

        df["ptype"] = df["ptype"].map(names).astype("category")

        return df, make_component_styles(names)
