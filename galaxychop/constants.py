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
>>> constants.plot_config.cold_disk.plot_color
'#4477AA'
>>> constants.plot_config.disk.plot_linestyle
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
class _ComponentConf:
    """
    Immutable dataclass to store the configuration of a single component.

    This class tracks all its instances via a `weakref.WeakSet` to dynamically
    determine the parent-child relationships between components.

    Parameters
    ----------
    label : str
        The human-readable name of the component (e.g., "Cold Disk").
    parent : _ComponentConf or None
        A direct reference to the parent component object, if any.
        This creates a hierarchy.
    plot_color : str
        The hexadecimal color code to be used for plotting this component.
        Main components use black, subcomponents use Paul Tol colors.
    plot_alpha : float
        The alpha (transparency) value to be used for plotting.
    plot_zorder : int
        The drawing order for plots (higher numbers are drawn on top).
    plot_linestyle : str
        The line style for plotting (matplotlib format).
        Main components have unique line styles for maximum accessibility.
    plot_linewidth : float
        The line width for plotting, scaled by component importance.

    """

    label: str
    parent: Optional["_ComponentConf"]
    plot_color: str
    plot_alpha: float
    plot_zorder: int
    plot_linestyle: str
    plot_linewidth: float

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
            "color": self.plot_color,
            "alpha": self.plot_alpha,
            "linestyle": self.plot_linestyle,
            "linewidth": self.plot_linewidth,
        }


# Main components use a fixed look (unique line styles for maximum
# accessibility). Subcomponents use Paul Tol Bright colors for
# differentiation within families.

# Base components
_no_component = _ComponentConf(
    label="Unclassified",
    parent=None,
    plot_color="#BBBBBB",  # Neutral gray
    plot_alpha=1.0,
    plot_zorder=0,
    plot_linestyle=":",  # Dotted for uncertain/unclassified
    plot_linewidth=1.0,
)

# MAIN COMPONENTS
# dm/gas/stars keep the fixed look GalaxyPlotter has always used, so
# existing galaxy plots don't change when they start reading it from here.
# They are drawn slightly thicker than every component below, and only
# gas/dm carry some alpha (making them read as slightly lighter); stars and
# every component are fully opaque.
_dm = _ComponentConf(
    label="Dark Matter",
    parent=None,
    plot_color="#222222",  # matches GalaxyPlotter's fixed look
    plot_alpha=0.7,
    plot_zorder=1,
    plot_linestyle=":",  # Dotted - most fragmented
    plot_linewidth=2.6,
)

_gas = _ComponentConf(
    label="Gas",
    parent=None,
    plot_color="tab:blue",  # matches GalaxyPlotter's fixed look
    plot_alpha=0.7,
    plot_zorder=2,
    plot_linestyle="--",  # Dashed - matches GalaxyPlotter's fixed look
    plot_linewidth=2.6,
)

_stars = _ComponentConf(
    label="Stars",
    parent=None,
    plot_color="tab:red",  # matches GalaxyPlotter's fixed look
    plot_alpha=1.0,
    plot_zorder=3,
    plot_linestyle="-",  # Solid - most continuous
    plot_linewidth=2.6,
)

_disk = _ComponentConf(
    label="Disk",
    parent=None,
    plot_color="#2E5994",  # Dark blue - darker than cold_disk
    plot_alpha=1.0,
    plot_zorder=4,
    plot_linestyle="-",  # Solid, same as stars (disk is a stellar structure)
    plot_linewidth=2.2,
)

# SUBCOMPONENTS (Paul Tol Bright colors)
_spheroid = _ComponentConf(
    label="Spheroid",
    parent=_stars,
    plot_color="#EE6677",  # Red (Paul Tol) - hot kinematic component
    plot_alpha=1.0,
    plot_zorder=3,
    plot_linestyle="-",  # Solid for spheroids
    plot_linewidth=1.8,
)

_cold_disk = _ComponentConf(
    label="Cold Disk",
    parent=_disk,
    plot_color="#4477AA",  # Blue (Paul Tol) - child of dark blue disk
    plot_alpha=1.0,
    plot_zorder=7,
    plot_linestyle="-",  # Solid for primary disk component
    plot_linewidth=2.0,
)

_warm_disk = _ComponentConf(
    label="Warm Disk",
    parent=_disk,
    plot_color="#228833",  # Green (Paul Tol) - intermediate component
    plot_alpha=1.0,
    plot_zorder=6,
    plot_linestyle="-",  # Solid for secondary disk component
    plot_linewidth=1.5,
)

_bar = _ComponentConf(
    label="Bar",
    parent=_disk,
    plot_color="#EE7733",  # Orange (Paul Tol) - prominent feature
    plot_alpha=1.0,
    plot_zorder=8,
    plot_linestyle="-",  # Solid for observed bar signatures
    plot_linewidth=2.5,
)

_bulge = _ComponentConf(
    label="Bulge",
    parent=_spheroid,
    plot_color="#EE6677",  # Red (Paul Tol) - same as parent spheroid
    plot_alpha=1.0,
    plot_zorder=5,
    plot_linestyle="-",  # Solid for classical bulge
    plot_linewidth=1.8,
)

_halo = _ComponentConf(
    label="Halo",
    parent=_spheroid,
    plot_color="#BBBBBB",  # Gray (Paul Tol) - diffuse component
    plot_alpha=1.0,
    plot_zorder=4,
    plot_linestyle="-",  # Solid for diffuse component
    plot_linewidth=1.2,
)


# Final configuration object.
# This Bunch instance is the single source of truth for component
# configurations and should be imported by other modules.
plot_config = bunch.Bunch(
    "galaxychop_config",
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
    },
)

#: Drawing order of the galaxy particle types, derived from their
#: ``plot_zorder`` in ``plot_config`` (stars go last, so they sit on top
#: of the more diffuse components).
PLOT_ORDER = tuple(
    sorted(
        plot_config.keys(),
        key=lambda ptype: plot_config[ptype].plot_zorder,
    )
)
