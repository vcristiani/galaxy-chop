# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Utilities module."""

# =============================================================================
# IMPORTS
# =============================================================================

from ._align import is_star_aligned, star_align
from ._center import center, is_centered
from ._circ import DEFAULT_CBIN, JCirc, jcirc
from ._decorators import doc_inherit
from ._potential import (
    DEFAULT_POTENTIAL_BACKEND,
    G,
    POTENTIAL_BACKENDS,
    potential,
)

__all__ = [
    "potential",
    "POTENTIAL_BACKENDS",
    "DEFAULT_POTENTIAL_BACKEND",
    "G",
    "center",
    "is_centered",
    "star_align",
    "is_star_aligned",
    "JCirc",
    "jcirc",
    "DEFAULT_CBIN",
    "doc_inherit",
]
