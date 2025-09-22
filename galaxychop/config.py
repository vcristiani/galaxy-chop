# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================


"""
Galaxy-chop default configuration with hierarchical color convention.

This module provides a centralized configuration for all the components
identified within a galaxy. It defines their properties (like labels, plotting
colors, line styles) and their hierarchical relationships, using a mixed
approach: black for main components and Paul Tol colors for subcomponents.

The main object, `config`, is a `Bunch` instance that grants
attribute-style access to the configuration of each component.

Examples
--------
To access the configuration of a component:

>>> from galaxychop import config
>>> config.config.dm.label
'Dark Matter'
>>> config.config.cold_disk.plot_color
'#4477AA'
>>> config.config.disk.plot_linestyle
'--'

To find the children of a component:

>>> config.config.disk.childrens
frozenset({<_ComponentConf bar>, <_ComponentConf cold_disk>,
<_ComponentConf warm_disk>})

"""

# =============================================================================
# IMPORTS
# =============================================================================

import dataclasses
from typing import Optional
import weakref

from .utils import bunch

# =============================================================================
# CONFIGURATION
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
        return {
            "color": self.plot_color,
            "alpha": self.plot_alpha,
            "linestyle": self.plot_linestyle,
            "linewidth": self.plot_linewidth,
        }


# =============================================================================
# HIERARCHICAL COLOR CONVENTION
# =============================================================================

# Main components use BLACK with unique line styles for maximum accessibility
# Subcomponents use Paul Tol Bright colors for differentiation within families

# Base components
_no_component = _ComponentConf(
    label="Unclassified",
    parent=None,
    plot_color="#BBBBBB",  # Neutral gray
    plot_alpha=0.2,
    plot_zorder=0,
    plot_linestyle=":",  # Dotted for uncertain/unclassified
    plot_linewidth=1.0,
)

# MAIN COMPONENTS (BLACK with unique line styles)
_dm = _ComponentConf(
    label="Dark Matter",
    parent=None,
    plot_color="#000000",  # BLACK - main component
    plot_alpha=0.7,
    plot_zorder=1,
    plot_linestyle=":",  # Dotted - most fragmented
    plot_linewidth=2.0,
)

_gas = _ComponentConf(
    label="Gas",
    parent=None,
    plot_color="#000000",  # BLACK - main component
    plot_alpha=0.7,
    plot_zorder=2,
    plot_linestyle="-.",  # Dash-dot - intermediate
    plot_linewidth=2.0,
)

_stars = _ComponentConf(
    label="Stars",
    parent=None,
    plot_color="#000000",  # BLACK - main component
    plot_alpha=1.0,
    plot_zorder=3,
    plot_linestyle="-",  # Solid - most continuous
    plot_linewidth=2.0,
)

_disk = _ComponentConf(
    label="Disk",
    parent=None,
    plot_color="#2E5994",  # Dark blue - darker than cold_disk
    plot_alpha=0.8,
    plot_zorder=4,
    plot_linestyle="--",  # Dashed - between dash-dot and solid
    plot_linewidth=2.2,
)

# SUBCOMPONENTS (Paul Tol Bright colors)
_spheroid = _ComponentConf(
    label="Spheroid",
    parent=_stars,
    plot_color="#EE6677",  # Red (Paul Tol) - hot kinematic component
    plot_alpha=0.5,
    plot_zorder=3,
    plot_linestyle="-",  # Solid for spheroids
    plot_linewidth=1.8,
)

_cold_disk = _ComponentConf(
    label="Cold Disk",
    parent=_disk,
    plot_color="#4477AA",  # Blue (Paul Tol) - child of dark blue disk
    plot_alpha=0.8,
    plot_zorder=7,
    plot_linestyle="-",  # Solid for primary disk component
    plot_linewidth=2.0,
)

_warm_disk = _ComponentConf(
    label="Warm Disk",
    parent=_disk,
    plot_color="#228833",  # Green (Paul Tol) - intermediate component
    plot_alpha=0.6,
    plot_zorder=6,
    plot_linestyle="-",  # Solid for secondary disk component
    plot_linewidth=1.5,
)

_bar = _ComponentConf(
    label="Bar",
    parent=_disk,
    plot_color="#EE7733",  # Orange (Paul Tol) - prominent feature
    plot_alpha=0.7,
    plot_zorder=8,
    plot_linestyle="-",  # Solid for observed bar signatures
    plot_linewidth=2.5,
)

_bulge = _ComponentConf(
    label="Bulge",
    parent=_spheroid,
    plot_color="#EE6677",  # Red (Paul Tol) - same as parent spheroid
    plot_alpha=0.6,
    plot_zorder=5,
    plot_linestyle="-",  # Solid for classical bulge
    plot_linewidth=1.8,
)

_halo = _ComponentConf(
    label="Halo",
    parent=_spheroid,
    plot_color="#BBBBBB",  # Gray (Paul Tol) - diffuse component
    plot_alpha=0.3,
    plot_zorder=4,
    plot_linestyle="-",  # Solid for diffuse component
    plot_linewidth=1.2,
)


# Final configuration object.
# This Bunch instance is the single source of truth for component
# configurations and should be imported by other modules.
config = bunch.Bunch(
    "galaxychop_config",
    {
        "no_component": _no_component,
        "dm": _dm,
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
