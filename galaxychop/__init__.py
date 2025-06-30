# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""
GalaxyChop.

Implementation of a few galaxy dynamic decomposition methods.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from . import constants, models, preproc, utils
from .core import (
    Galaxy,
    NoGravitationalPotentialError,
    ParticleSet,
    ParticleSetType,
    mkgalaxy,
)
from .io import read_hdf5, to_hdf5
from .pipeline import GchopPipeline, mkpipe


__version__ = tuple(constants.VERSION.split("."))

__all__ = [
    "Galaxy",
    "ParticleSet",
    "ParticleSetType",
    "NoGravitationalPotentialError",
    "io",
    "models",
    "preproc",
    "utils",
    "mkgalaxy",
    "read_hdf5",
    "to_hdf5",
    "constants",
    "GchopPipeline",
    "mkpipe",
]
