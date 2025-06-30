# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""utils module."""

# =============================================================================
# IMPORTS
# =============================================================================
from .bunch import Bunch
from .decorators import doc_inherit
from .unames import unique_names

__all__ = [
    "doc_inherit",
    "Bunch",
    "unique_names",
]
