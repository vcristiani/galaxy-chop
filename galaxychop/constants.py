# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""
Constants for use inside galaxychop.

This module also provides a centralized configuration for all the
components identified within a galaxy. It defines their properties (like
labels, plotting colors, line styles) and their hierarchical relationships,
using a mixed approach: a fixed look for the main stars/gas/dark_matter
components and Paul Tol colors for subcomponents.

The object ``plot_config`` is a `Bunch` instance that grants attribute-style
access to the configuration of each component.

Examples
--------
To access the configuration of a component:

>>> from galaxychop import constants
>>> constants.plot_config.dark_matter.label
'Dark Matter'
>>> constants.plot_config.cold_disk.color
'#4477AA'
>>> constants.plot_config.disk.linestyle
'-'

To find the children of a component:

>>> sorted(comp.label for comp in constants.plot_config.disk.childrens)
['Bar', 'Cold Disk', 'Warm Disk']

"""

# =============================================================================
# IMPORTS
# =============================================================================

import dataclasses
import weakref
from importlib.metadata import version
from typing import Optional

from astropy import constants as c
from astropy import units as u

from .utils import bunch

# =============================================================================
# PROJECT VERSION
# =============================================================================

NAME = "galaxychop"

VERSION = version(NAME)

# =============================================================================
# STELLAR DYNAMICS
# =============================================================================

SD_DEFAULT_CBIN = (0.05, 0.005)
"""
Default binning of circularity for stellar dynamics calculation.

Please check the documentation of ``galaxychop.circ.stellar_dynamics()``.

"""
SD_DEFAULT_REASSIGN = False
"""
Default value to reassign the values of the particle stellar dynamics.

Please check the documentation of ``galaxychop.circ.stellar_dynamics()``.

"""

SD_RUNTIME_WARNING_ACTION = "ignore"
"""
Default of "what-to-do" about the RuntimeWarning in stellar_dynamics \
calculation.

Please check the documentation of ``galaxychop.circ.stellar_dynamics()``.

"""

# =============================================================================
# Gravity
# =============================================================================

#: GalaxyChop Gravitational unit
G_UNIT = (u.km**2 * u.kpc) / (u.s**2 * u.solMass)

#: Gravitational constant as float in G_UNIT
G = c.G.to(G_UNIT).to_value()

# =============================================================================
# GALAXY PLOT COMPONENTS CONFIGURATION
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _ComponentPlotStyle:
    """
    Immutable dataclass to store the plotting style of a single component.

    Field names match the matplotlib/seaborn kwargs they end up feeding
    (``color``, ``alpha``, ``linestyle``, ``linewidth``) so a style can be
    propagated with a plain ``**get_mplstyle()`` instead of a hand-written
    translation at every call site.

    This class tracks all its instances via a `weakref.WeakSet` to dynamically
    determine the parent-child relationships between components.

    Parameters
    ----------
    label : str
        The human-readable name of the component (e.g., "Cold Disk").
    parent : _ComponentPlotStyle or None
        A direct reference to the parent component object, if any.
        This creates a hierarchy.
    color : str
        The hexadecimal color code to be used for plotting this component.
        Main components use black, subcomponents use Paul Tol colors.
    alpha : float
        The alpha (transparency) value to be used for plotting.
    zorder : int
        The drawing order for plots (higher numbers are drawn on top).
    linestyle : str
        The line style for plotting (matplotlib format).
        Main components have unique line styles for maximum accessibility.
    linewidth : float
        The line width for plotting, scaled by component importance.

    """

    label: str
    parent: Optional["_ComponentPlotStyle"]
    color: str
    alpha: float
    zorder: int
    linestyle: str
    linewidth: float

    _instances = weakref.WeakSet()

    def __post_init__(self):
        """Register the instance in the weakset after creation."""
        self._instances.add(self)

    @property
    def childrens(self):
        """Return a frozenset of all direct children of this component."""
        return frozenset(
            child for child in self._instances if child.parent is self
        )

    def get_mplstyle(self):
        """Return the matplotlib style kwargs for this component."""
        return {
            "color": self.color,
            "alpha": self.alpha,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
        }


# Main components use a fixed look (unique line styles for maximum
# accessibility). Subcomponents use Paul Tol Bright colors for
# differentiation within families.

# WHOLE GALAXY
# The pooled curve (e.g. GalaxyPlotter.rotation_curve's galaxy=True line).
# It is the root of every other component and is plotted first, right
# before the unclassified bucket, so everything else is drawn over it.
_galaxy = _ComponentPlotStyle(
    label="Galaxy",
    parent=None,
    color="black",  # matches GalaxyPlotter's fixed look
    alpha=1.0,
    zorder=-1,
    linestyle="-",  # Solid - matches GalaxyPlotter's fixed look
    linewidth=2.5,  # same weight as the main particle types
)

# Base components
_no_component = _ComponentPlotStyle(
    label="Unclassified",
    parent=_galaxy,
    color="#BBBBBB",  # Neutral gray
    alpha=1.0,
    zorder=0,
    linestyle=":",  # Dotted for uncertain/unclassified
    linewidth=1.0,
)

# MAIN COMPONENTS
# dm/gas/stars keep the fixed look GalaxyPlotter has always used, so
# existing galaxy plots don't change when they start reading it from here.
# They are drawn slightly thicker than every component below, and only
# gas/dm carry some alpha (making them read as slightly lighter); stars and
# every component are fully opaque.
_dm = _ComponentPlotStyle(
    label="Dark Matter",
    parent=_galaxy,
    color="#222222",  # matches GalaxyPlotter's fixed look
    alpha=0.5,
    zorder=1,
    linestyle=":",  # Dotted - most fragmented
    linewidth=2,
)

_gas = _ComponentPlotStyle(
    label="Gas",
    parent=_galaxy,
    color="tab:blue",  # matches GalaxyPlotter's fixed look
    alpha=0.5,
    zorder=2,
    linestyle="--",  # Dashed - matches GalaxyPlotter's fixed look
    linewidth=2,
)

_stars = _ComponentPlotStyle(
    label="Stars",
    parent=_galaxy,
    color="tab:red",  # matches GalaxyPlotter's fixed look
    alpha=1.0,
    zorder=3,
    linestyle="-",  # Solid - most continuous
    linewidth=2,
)

_disk = _ComponentPlotStyle(
    label="Disk",
    parent=_galaxy,
    color="#2E5994",  # Dark blue - darker than cold_disk
    alpha=1.0,
    zorder=4,
    linestyle="-",  # Solid, same as stars (disk is a stellar structure)
    linewidth=1,
)

# SUBCOMPONENTS (Paul Tol Bright colors)
_spheroid = _ComponentPlotStyle(
    label="Spheroid",
    parent=_stars,
    color="#EE6677",  # Red (Paul Tol) - hot kinematic component
    alpha=1.0,
    zorder=3,
    linestyle="-",  # Solid for spheroids
    linewidth=1,
)

_cold_disk = _ComponentPlotStyle(
    label="Cold Disk",
    parent=_disk,
    color="#4477AA",  # Blue (Paul Tol) - child of dark blue disk
    alpha=1.0,
    zorder=7,
    linestyle="-",  # Solid for primary disk component
    linewidth=1,
)

_warm_disk = _ComponentPlotStyle(
    label="Warm Disk",
    parent=_disk,
    color="#228833",  # Green (Paul Tol) - intermediate component
    alpha=1.0,
    zorder=6,
    linestyle="-",  # Solid for secondary disk component
    linewidth=1,
)

_bar = _ComponentPlotStyle(
    label="Bar",
    parent=_disk,
    color="#EE7733",  # Orange (Paul Tol) - prominent feature
    alpha=1.0,
    zorder=8,
    linestyle="-",  # Solid for observed bar signatures
    linewidth=1,
)

_bulge = _ComponentPlotStyle(
    label="Bulge",
    parent=_spheroid,
    color="#EE6677",  # Red (Paul Tol) - same as parent spheroid
    alpha=1.0,
    zorder=5,
    linestyle="-",  # Solid for classical bulge
    linewidth=1,
)

_halo = _ComponentPlotStyle(
    label="Halo",
    parent=_spheroid,
    color="#BBBBBB",  # Gray (Paul Tol) - diffuse component
    alpha=1.0,
    zorder=4,
    linestyle="-",  # Solid for diffuse component
    linewidth=1,
)


# Final configuration object.
# This Bunch instance is the single source of truth for component
# configurations and should be imported by other modules.
plot_config = bunch.Bunch(
    "plot_config",
    {
        "no_component": _no_component,
        "dark_matter": _dm,
        "gas": _gas,
        "stars": _stars,
        "disk": _disk,
        "spheroid": _spheroid,
        "cold_disk": _cold_disk,
        "warm_disk": _warm_disk,
        "bar": _bar,
        "bulge": _bulge,
        "halo": _halo,
        "galaxy": _galaxy,
    },
)

# The components only need to be reachable through plot_config from here
# on; drop the module-level names so plot_config stays the single source
# of truth.
del (
    _galaxy,
    _no_component,
    _dm,
    _gas,
    _stars,
    _disk,
    _spheroid,
    _cold_disk,
    _warm_disk,
    _bar,
    _bulge,
    _halo,
)

#: Drawing order of the galaxy particle types, derived from their
#: ``zorder`` in ``plot_config`` (stars go last, so they sit on top
#: of the more diffuse components).
PLOT_ORDER = tuple(
    sorted(
        plot_config.keys(),
        key=lambda ptype: plot_config[ptype].zorder,
    )
)


def make_component_styles(names):
    """
    Build the fixed plot style for the given components.

    Parameters
    ----------
    names : dict
        Maps each component key in ``plot_config`` to its display name,
        in drawing order.

    Returns
    -------
    dict
        Keys ``hue_order``, ``palette``, ``linestyles``, ``alphas`` and
        ``linewidths``, indexed by the display names of the components.
    """
    return {
        "hue_order": list(names.values()),
        "palette": {names[p]: plot_config[p].color for p in names},
        "linestyles": {names[p]: plot_config[p].linestyle for p in names},
        "alphas": {names[p]: plot_config[p].alpha for p in names},
        "linewidths": {names[p]: plot_config[p].linewidth for p in names},
    }
