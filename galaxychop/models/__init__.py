# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Module for dynamical decomposition models."""

# =============================================================================
# IMPORTS
# =============================================================================

from .core import (
    DecomposedParticleSet,
    DecomposedGalaxy,
    GalaxyDecomposerABC,
    hparam,
)
from .gaussian_mixture import (
    AutoGaussianMixture,
    DynamicStarsGaussianDecomposerABC,
    GaussianMixture,
)
from .histogram import JEHistogram, JHistogram
from .kmeans import KMeans
from .threshold import JThreshold


# =============================================================================
# MAKE IT PUBLIC!
# =============================================================================

__all__ = [
    "DecomposedGalaxy",
    "GalaxyDecomposerABC",
    "DecomposedParticleSet",
    "JThreshold",
    "JHistogram",
    "JEHistogram",
    "KMeans",
    "DynamicStarsGaussianDecomposerABC",
    "GaussianMixture",
    "AutoGaussianMixture",
    "hparam",
]
