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
# META
# =============================================================================

__version__ = "0.3.dev0"


# =============================================================================
# IMPORTS
# =============================================================================

from . import io, models, preproc, utils
from .core import (
    Galaxy,
    ParticleSet,
    ParticleSetType,
    mkgalaxy,
)


__all__ = [
    "Galaxy",
    "ParticleSet",
    "ParticleSetType",
    "io",
    "models",
    "preproc",
    "utils",
    "mkgalaxy",
]
