# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Utilities module."""

# =============================================================================
# IMPORTS
# =============================================================================

from ._potential import (
    potential,
    G,
    POTENTIAL_BACKENDS,
    DEFAULT_POTENTIAL_BACKEND,
)
from ._center import center, is_centered
from ._align import star_align, is_star_aligned
from ._circ import jcirc, DEFAULT_CBIN
from ._decorators import doc_inherit


__all__ = [
    "potential",
    "POTENTIAL_BACKENDS",
    "DEFAULT_POTENTIAL_BACKEND",
    "G",
    "center",
    "is_centered",
    "star_align",
    "is_star_aligned",
    "jcirc",
    "DEFAULT_CBIN",
    "doc_inherit",
]
