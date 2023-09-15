# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Base objects and functions og galaxychop."""

# =============================================================================
# IMPORTS
# =============================================================================

from .data import (
    Galaxy,
    ParticleSet,
    ParticleSetType,
    mkgalaxy,
)
from . import plot, sdynamics


__all__ = [
    "Galaxy",
    "ParticleSet",
    "ParticleSetType",
    "mkgalaxy",
    "plot",
    "sdynamics",
]
