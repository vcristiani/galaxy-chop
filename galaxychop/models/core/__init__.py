# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Module for decomposed galaxies."""

# =============================================================================
# IMPORTS
# =============================================================================

from .decomposed_galaxy import (
    DecomposedParticleSet,
    DecomposedGalaxy,
)
from .galaxy_decomposer_abc import GalaxyDecomposerABC, hparam

__all__ = [
    "DecomposedParticleSet",
    "DecomposedGalaxy",
    "GalaxyDecomposerABC",
    "hparam",
]
