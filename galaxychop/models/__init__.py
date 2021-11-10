# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module for dynamical decomposition models."""


# =============================================================================
# IMPORTS
# =============================================================================

from ._base import DynamicStarsDecomposerMixin, GalaxyDecomposerABC, hparam
from ._gaussian_mixture import (
    AutoGaussianMixture,
    GaussianABC,
    GaussianMixture,
)
from ._kmeans import KMeans
from ._threshold import JThreshold

# from ._histogram import JEHistogram, JHistogram


# =============================================================================
# MAKE IT PUBLIC!
# =============================================================================

__all__ = [
    "GalaxyDecomposerABC",
    "DynamicStarsDecomposerMixin",
    "JThreshold",
    "JHistogram",
    "JEHistogram",
    "KMeans",
    "GaussianABC",
    "GaussianMixture",
    "AutoGaussianMixture",
    "hparam",
]
