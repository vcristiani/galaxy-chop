# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

import galaxychop as gchop

import numpy as np

# import pandas as pd

import pytest

# =============================================================================
# DECOMPOSER ABC
# =============================================================================


@pytest.mark.model
def test_GalaxyDecomposerABC_not_implemented():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            return super().get_attributes()

        def split(self, X, y, attributes):
            return super().split(X, y, attributes)

    decomposer = Decomposer()

    with pytest.raises(NotImplementedError):
        decomposer.get_attributes()

    with pytest.raises(NotImplementedError):
        decomposer.split(None, None, None)


@pytest.mark.model
@pytest.mark.parametrize(
    "bins_value", [None, (1.0,), (1.0, 2.0, 3.0), (1.0, 2)]
)
def test_GalaxyDecomposerABC_invalid_bins(bins_value):
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            ...

        def split(self, X, y, attributes):
            ...

    with pytest.raises(ValueError):
        Decomposer(cbins=bins_value)


@pytest.mark.model
def test_GalaxyDecomposerABC_repr():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        other = gchop.models.hparam(default=1)

        def get_attributes(self):
            return ["normalized_star_energy", "eps", "eps_r"]

        def split(self, X, y, attributes):
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
        def get_attributes(self):
            return ["eps"]

        def split(self, X, y, attributes):
            return np.full(len(X), 100), None

    decomposer = Decomposer()

    gal_decomp = decomposer.decompose(gal)

    assert len(gal_decomp) == len(gal)
    assert isinstance(gal_decomp, gchop.models.DecomposedGalaxy)
    assert gal_decomp.method == "Decomposer"


@pytest.mark.model
def test_get_valid_stellar_mask():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            return ["eps"]

        def split(self, X, y, attributes):
            return np.array([0]), None

    dec = Decomposer()
    X = np.array([[1.0], [np.nan]])
    y = np.array([gchop.core.ParticleSetType.STARS.value,
                  gchop.core.ParticleSetType.STARS.value])

    mask = dec.get_valid_stellar_mask(X, y, ["eps"])
    assert mask.shape == (2,)
    assert mask.sum() == 1


@pytest.mark.model
def test_assign_components_and_probabilities():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            return ["eps"]

        def split(self, X, y, attributes):
            return np.array([1, 2]), np.array([[0.1, 0.9], [0.8, 0.2]])

    dec = Decomposer()
    X = np.ones((2, 1))
    valid_mask = np.array([True, True])

    comp = dec.assign_components_to_all_particles(
        X,
        np.array([1, 2]), valid_mask
    )
    assert np.array_equal(comp, [1, 2])

    probs = dec.assign_probabilities_to_all_particles(
        X,
        np.array([[0.1, 0.9], [0.8, 0.2]]), valid_mask
    )
    assert probs.shape == (2, 2)

    probs_none = dec.assign_probabilities_to_all_particles(X, None, valid_mask)
    assert np.isnan(probs_none).all()
