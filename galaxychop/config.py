# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================


"""
Galaxy-chop default configuration.

This module provides a centralized configuration for all the components
identified within a galaxy. It defines their properties (like labels and
plotting colors) and their hierarchical relationships.

The main object, `config`, is a `Bunch` instance that grants
attribute-style access to the configuration of each component.

Examples
--------
To access the configuration of a component:

>>> from galaxychop import config
>>> config.config.dm.label
'Dark Matter'
>>> config.config.cold_disk.plot_color
'#1f77b4'

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
    plot_alpha : float
        The alpha (transparency) value to be used for plotting.
    plot_zorder : int
        The drawing order for plots (higher numbers are drawn on top).

    """

    label: str
    parent: Optional["_ComponentConf"]
    plot_color: str
    plot_alpha: float
    plot_zorder: int

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


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# To handle the self-referential `parent` attribute in a frozen dataclass,
# we define the components and then link them.

# Base components
_no_component = _ComponentConf(
    label="Unclassified",
    parent=None,
    plot_color="#d3d3d3",
    plot_alpha=0.2,
    plot_zorder=0,
)
_dm = _ComponentConf(
    label="Dark Matter",
    parent=None,
    plot_color="#7f7f7f",
    plot_alpha=0.3,
    plot_zorder=1,
)
_gas = _ComponentConf(
    label="Gas",
    parent=None,
    plot_color="#d62728",
    plot_alpha=0.5,
    plot_zorder=2,
)
_stars = _ComponentConf(
    label="Stars",
    parent=None,
    plot_color="#ff7f0e",
    plot_alpha=1.0,
    plot_zorder=3,
)

# Stellar components with parent-child relationships
_disk = _ComponentConf(
    label="Disk",
    parent=_stars,
    plot_color="#aec7e8",
    plot_alpha=0.7,
    plot_zorder=4,
)
_spheroid = _ComponentConf(
    label="Spheroid",
    parent=_stars,
    plot_color="#c7c7c7",
    plot_alpha=0.5,
    plot_zorder=3,
)
_cold_disk = _ComponentConf(
    label="Cold Disk",
    parent=_disk,
    plot_color="#1f77b4",
    plot_alpha=0.8,
    plot_zorder=7,
)
_warm_disk = _ComponentConf(
    label="Warm Disk",
    parent=_disk,
    plot_color="#17becf",
    plot_alpha=0.7,
    plot_zorder=6,
)
_bar = _ComponentConf(
    label="Bar",
    parent=_disk,
    plot_color="#2ca02c",
    plot_alpha=0.7,
    plot_zorder=8,
)
_bulge = _ComponentConf(
    label="Bulge",
    parent=_spheroid,
    plot_color="#e377c2",
    plot_alpha=0.6,
    plot_zorder=5,
)
_halo = _ComponentConf(
    label="Halo",
    parent=_spheroid,
    plot_color="#9467bd",
    plot_alpha=0.4,
    plot_zorder=4,
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
