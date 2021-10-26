# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module for dynamical decomposition models."""


# =============================================================================
# IMPORTS
# =============================================================================

from ._base import GalaxyDecomposerABC, DynamicStarsDecomposerMixin, hparam

# from ._gaussian_mixture import AutoGaussianMixture, GaussianMixture
# from ._histogram import JEHistogram, JHistogram
from ._kmeans import KMeans

# from ._threshold import JThreshold

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
    "GaussianMixture",
    "AutoGaussianMixture",
    "hparam",
]
