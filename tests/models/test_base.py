import galaxychop as gchop
from galaxychop import models

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# COMPONENTS
# =============================================================================


@pytest.mark.model
@pytest.mark.parametrize("probs", [True, False])
def test_Components(probs):
    random = np.random.default_rng(42)

    labels = random.integers(0, 3, 100)
    ptypes = np.ones(100)
    probabilities = random.normal(size=100) if probs else None

    components = models.Components(
        labels=labels, ptypes=ptypes, probabilities=probabilities
    )

    assert len(components) == 100
    assert (
        repr(components)
        == f"Components(100, labels=[0 1 2], probabilities={probs})"
    )


@pytest.mark.model
@pytest.mark.parametrize("probs", [True, False])
def test_Components_bad_len(probs):
    random = np.random.default_rng(42)

    labels = random.integers(0, 3, 100)
    ptypes = np.ones(99)
    probabilities = random.normal(size=98) if probs else None

    with pytest.raises(ValueError):
        models.Components(
            labels=labels, ptypes=ptypes, probabilities=probabilities
        )


@pytest.mark.model
@pytest.mark.parametrize("probs", [True, False])
def test_Components_to_dataframe(probs):
    random = np.random.default_rng(42)

    labels = random.integers(0, 3, 100)
    ptypes = np.ones(100)
    probabilities = random.normal(size=100) if probs else None

    components = models.Components(
        labels=labels, ptypes=ptypes, probabilities=probabilities
    )

    expected = pd.DataFrame({"labels": labels, "ptypes": ptypes})

    if probs:
        probs_df = pd.DataFrame({"probs_0": probabilities})
        expected = pd.concat([expected, probs_df], axis=1)

    pd.testing.assert_frame_equal(components.to_dataframe(), expected)


# =============================================================================
# DECOMPOSER ABC
# =============================================================================


@pytest.mark.model
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


@pytest.mark.model
@pytest.mark.parametrize(
    "bins_value", [None, (1.0,), (1.0, 2.0, 3.0), (1.0, 2)]
)
def test_GalaxyDecomposerABC_invalid_bins(bins_value):
    class Decomposer(models.GalaxyDecomposerABC):
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
    class Decomposer(models.GalaxyDecomposerABC):

        other = models.hparam(default=1)

        def get_attributes(self):
            return ["normalized_star_energy", "eps", "eps_r"]

        def split(self, X, y, attributes):
            ...

        def get_rows_mask(self, X, y, attributes):
            ...

    decomposer = Decomposer(cbins=(0.3, 0.2), other="zaraza")
    result = repr(decomposer)
    expected = "Decomposer(cbins=(0.3, 0.2), other='zaraza')"

    assert result == expected


@pytest.mark.model
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
    jcirc = gchop.jcirc(gal)

    X_stars = X[t == gchop.ParticleSetType.STARS.value]
    assert np.array_equal(X_stars[:, 1], jcirc.eps, equal_nan=True)

    X_nostars = X[t != gchop.ParticleSetType.STARS.value]
    assert np.all(np.isnan(X_nostars[:, 1]))


@pytest.mark.model
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


@pytest.mark.model
def test_GalaxyDecomposerABC_decompose(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    class Decomposer(models.GalaxyDecomposerABC):
        def get_attributes(self):
            return ["x"]

        def split(self, X, y, attributes):
            return np.full(len(X), 100), None

        def get_rows_mask(self, X, y, attributes):
            return y == 2

    decomposer = Decomposer()

    components = decomposer.decompose(gal)

    assert (components.ptypes == "stars").sum() == len(gal.stars)
    assert (components.ptypes == "dark_matter").sum() == len(gal.dark_matter)
    assert (components.ptypes == "gas").sum() == len(gal.gas)

    assert np.all(components.labels[components.ptypes == "gas"] == 100)
    assert np.all(np.isnan(components.labels[components.ptypes != "gas"]))


# =============================================================================
# DYNAMIC STARS DECOMPOSER
# =============================================================================


@pytest.mark.model
def test_DynamicStarDecomposer_get_attributes():
    class Decomposer(
        models.DynamicStarsDecomposerMixin,
        models.GalaxyDecomposerABC,
    ):
        def get_attributes(self):
            return ["normalized_star_energy", "eps", "eps_r"]

        def split(self, X, y, attributes):
            ...

    decomposer = Decomposer()

    assert decomposer.get_attributes() == [
        "normalized_star_energy",
        "eps",
        "eps_r",
    ]


@pytest.mark.model
def test_DynamicStarDecomposer_get_rows_mask():
    class Decomposer(
        models.DynamicStarsDecomposerMixin,
        models.GalaxyDecomposerABC,
    ):
        def get_attributes(self):
            return ["normalized_star_energy", "eps", "eps_r"]

        def split(self, X, y, attributes):
            ...

    decomposer = Decomposer()

    X = [[1, 2, 3], [np.nan, 2, 3], [1, 2, np.nan], [1, 2, 3]]

    y = [0, 0, 1, 1]

    attrs = ["a", "b", "c"]
    result = decomposer.get_rows_mask(X, y, attrs)

    assert np.all(result == [True, False, False, False])
