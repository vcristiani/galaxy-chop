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

from .abc import (
    GalaxyDecomposerABC,
    hparam,
)
from .dgalaxy import  DecomposedGalaxy
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
    "Components",
    "GalaxyDecomposerABC",
    "JThreshold",
    "JHistogram",
    "JEHistogram",
    "KMeans",
    "DynamicStarsGaussianDecomposerABC",
    "GaussianMixture",
    "AutoGaussianMixture",
    "hparam",
]
