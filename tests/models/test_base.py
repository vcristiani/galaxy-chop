import warnings

import attr

import numpy as np

import pytest

import galaxychop as gchop
from galaxychop import models


def test_GalaxyDecomposerABC_not_implemethed():
    class Decomposer(models.GalaxyDecomposerABC):
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

    with pytest.raises(NotImplementedError):
        decomposer.get_rows_mask(None, None, None)


def test_GalaxyDecomposerABC_repr():
    class Decomposer(models.GalaxyDecomposerABC):

        other = models.hparam(default=1)

        def get_attributes(self):
            return ["normalized_star_energy", "eps", "eps_r"]

        def split(self, X, y, attributes):
            ...

        def get_rows_mask(self, X, y, attributes):
            ...

    decomposer = Decomposer(bins=(0.3, 0.2), other="zaraza")
    result = repr(decomposer)
    expected = "Decomposer(bins=(0.3, 0.2), other='zaraza')"

    assert result == expected


def test_GalaxyDecomposerABC_attributes_matrix(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    class Decomposer(models.GalaxyDecomposerABC):
        def get_attributes(self):
            ...

        def split(self, X, y, attributes):
            ...

        def get_rows_mask(self, X, y, attributes):
            ...

    decomposer = Decomposer()

    attributes = ["x", "eps"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X, t = decomposer.attributes_matrix(gal, attributes=attributes)

    # check types stars-dm-gas
    assert np.all(t[: len(gal.stars)] == gchop.ParticleSetType.STARS.value)
    assert np.all(
        t[len(gal.stars) : len(gal.dark_matter)]
        == gchop.ParticleSetType.DARK_MATTER.value
    )
    assert np.all(
        t[len(gal.stars) + len(gal.dark_matter) : len(gal.gas)]
        == gchop.ParticleSetType.GAS.value
    )

    # check as_dataframe attrs
    assert np.all(X[:, 0] == gal.to_dataframe(attributes=["x"])["x"])

    # check jcirc eps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        jcirc = gchop.jcirc(gal)

    X_stars = X[t == gchop.ParticleSetType.STARS.value]
    assert np.array_equal(X_stars[:, 1], jcirc.eps, equal_nan=True)

    X_nostars = X[t != gchop.ParticleSetType.STARS.value]
    assert np.all(np.isnan(X_nostars[:, 1]))


def test_GalaxyDecomposerABC_complete_labels():
    class Decomposer(models.GalaxyDecomposerABC):
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


def test_GalaxyDecomposerABC_decompose(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    class Decomposer(models.GalaxyDecomposerABC):
        def get_attributes(self):
            return ["x"]

        def split(self, X, y, attributes):
            return np.full(len(X), 100)

        def get_rows_mask(self, X, y, attributes):
            return y == 2

    decomposer = Decomposer()

    labels, y = decomposer.decompose(gal)

    assert (y == 0).sum() == len(gal.stars)
    assert (y == 1).sum() == len(gal.dark_matter)
    assert (y == 2).sum() == len(gal.gas)

    assert np.all(labels[y == 2] == 100)
    assert np.all(np.isnan(labels[y != 2]))
