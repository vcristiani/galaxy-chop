# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module sklearn models."""

# #####################################################
# IMPORTS
# #####################################################

from galaxychop import core

from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans

# #####################################################
# GCDecomposeMixin CLASS
# #####################################################


class GCDecomposeMixin:
    """Galaxy chop decompose mixin class."""

    def decompose(self, galaxy):
        """Decompose method."""
        if not isinstance(galaxy, core.Galaxy):
            found = type(galaxy)
            raise TypeError(
                f"'galaxy' must be a core.Galaxy instance. Found {found}"
            )

        X, y = galaxy.values()
        return self.fit_transform(X, y)


# #####################################################
# GCClusterMixin CLASS
# #####################################################


class GCClusterMixin(GCDecomposeMixin, ClusterMixin):
    """Galaxy chop cluster mixin class."""

    pass


# #####################################################
# GCKmeans CLASS
# #####################################################


class GCKmeans(GCClusterMixin, KMeans):
    """Galaxy chop KMean class."""

    pass
