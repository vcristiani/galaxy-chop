# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Base objects and functions of galaxychop."""

# =============================================================================
# IMPORTS
# =============================================================================

from . import plot, sdynamics
from .data import (
    Galaxy,
    NoGravitationalPotentialError,
    ParticleSet,
    ParticleSetType,
    mkgalaxy,
)
from .methods import GchopMethodABC


__all__ = [
    "Galaxy",
    "NoGravitationalPotentialError",
    "ParticleSet",
    "ParticleSetType",
    "mkgalaxy",
    "plot",
    "sdynamics",
    "GchopMethodABC",
]
