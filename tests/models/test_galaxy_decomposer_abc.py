# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

import galaxychop as gchop

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# DECOMPOSER ABC
# =============================================================================

@pytest.mark.model
def test_GalaxyDecomposerABC_not_implemented():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_stellar_attributes(self):
            return super().get_stellar_attributes()

        def split(self, X, y, stellar_properties):
            return super().split(X, y, stellar_properties)

    decomposer = Decomposer()

    with pytest.raises(NotImplementedError):
        decomposer.get_stellar_attributes()

    with pytest.raises(NotImplementedError):
        decomposer.split(None, None, None)

@pytest.mark.model
@pytest.mark.parametrize(
    "bins_value", [None, (1.0,), (1.0, 2.0, 3.0), (1.0, 2)]
)
def test_GalaxyDecomposerABC_invalid_bins(bins_value):
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_stellar_attributes(self):
            ...

        def split(self, X, y, stellar_properties):
            ...

    with pytest.raises(ValueError):
        Decomposer(cbins=bins_value)


@pytest.mark.model
def test_GalaxyDecomposerABC_repr():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        other = gchop.models.hparam(default=1)

        def get_stellar_attributes(self):
            return ["normalized_star_energy", "eps", "eps_r"]

        def split(self, X, y, stellar_properties):
            ...

    decomposer = Decomposer(cbins=(0.3, 0.2), reassign=True, other="zaraza")
    result = repr(decomposer)
    expected = "<Decomposer cbins=(0.3, 0.2), reassign=True, other='zaraza'>"

    assert result == expected

@pytest.mark.model
def test_GalaxyDecomposerABC_decompose(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_stellar_attributes(self):
            return ["eps"]

        def split(self, X, y, stellar_properties):
            return np.full(len(X), 100), None

    decomposer = Decomposer()

    gal_decomp = decomposer.decompose(gal)

    assert len(gal_decomp) == len(gal)
    assert isinstance(gal_decomp, gchop.models.DecomposedGalaxy)
    assert gal_decomp.method == "Decomposer"