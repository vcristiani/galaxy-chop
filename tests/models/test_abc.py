# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

import galaxychop as gchop

import numpy as np

import pytest

# =============================================================================
# DECOMPOSER ABC
# =============================================================================


@pytest.mark.model
def test_GalaxyDecomposerABC_not_implemethed():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            return super().get_attributes()

        def split(self, X, y, attributes):
            return super().split(X, y, attributes)

        def get_rows_mask(self, X, y, attributes):
            return super().get_rows_mask(X, y, attributes)

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

        def get_rows_mask(self, X, y, attributes):
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

        def get_rows_mask(self, X, y, attributes):
            ...

    decomposer = Decomposer(cbins=(0.3, 0.2), reassign=True, other="zaraza")
    result = repr(decomposer)
    expected = "Decomposer(cbins=(0.3, 0.2), reassign=True, other='zaraza')"

    assert result == expected


@pytest.mark.model
def test_GalaxyDecomposerABC_attributes_matrix(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            ...

        def split(self, X, y, attributes):
            ...

        def get_rows_mask(self, X, y, attributes):
            ...

    decomposer = Decomposer()

    attributes = ["x", "eps"]

    X, t = decomposer.attributes_matrix(gal, attributes=attributes)

    # check types stars-dm-gas
    assert np.all(t[: len(gal.stars)] == gchop.ParticleSetType.STARS.value)
    assert np.all(
        t[len(gal.stars) : len(gal.dark_matter)]  # noqa
        == gchop.ParticleSetType.DARK_MATTER.value
    )
    assert np.all(
        t[len(gal.stars) + len(gal.dark_matter) : len(gal.gas)]  # noqa
        == gchop.ParticleSetType.GAS.value
    )

    # check as_dataframe attrs
    assert np.all(X[:, 0] == gal.to_dataframe(attributes=["x"])["x"])

    # check jcirc eps
    jcirc = gal.stellar_dynamics()

    X_stars = X[t == gchop.ParticleSetType.STARS.value]
    assert np.array_equal(X_stars[:, 1], jcirc.eps, equal_nan=True)

    X_nostars = X[t != gchop.ParticleSetType.STARS.value]
    assert np.all(np.isnan(X_nostars[:, 1]))


@pytest.mark.model
def test_GalaxyDecomposerABC_complete_labels():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            ...

        def split(self):
            ...

        def get_rows_mask(self, X, y, attributes):
            ...

    decomposer = Decomposer()

    X = np.random.rand(3, 4)
    labels = [1, 1]
    rows_mask = [True, False, True]

    result = decomposer.complete_labels(
        X=X, labels=labels, rows_mask=rows_mask
    )

    assert np.array_equal(result, [1, np.nan, 1], equal_nan=True)


@pytest.mark.model
def test_GalaxyDecomposerABC_decompose(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            return ["x"]

        def split(self, X, y, attributes):
            return np.full(len(X), 100), None

        def get_rows_mask(self, X, y, attributes):
            return y == 2

    decomposer = Decomposer()

    gal_decomp = decomposer.decompose(gal)
    gal_components = gal_decomp.components

    assert (gal_components.ptypes == "stars").sum() == len(gal.stars)
    assert (gal_components.ptypes == "dark_matter").sum() == len(
        gal.dark_matter
    )
    assert (gal_components.ptypes == "gas").sum() == len(gal.gas)

    assert np.all(gal_components.labels[gal_components.ptypes == "gas"] == 100)
    assert np.all(
        np.isnan(gal_components.labels[gal_components.ptypes != "gas"])
    )
