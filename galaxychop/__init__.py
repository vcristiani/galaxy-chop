# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
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

from .data import *  # noqa
from . import io, models, preproc  # noqa


__all__ = ["io", "models", "preproc"]
