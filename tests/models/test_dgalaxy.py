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
# COMPONENTS
# =============================================================================


@pytest.mark.model
@pytest.mark.parametrize("probs", [True, False])
def test_Components(probs):
    random = np.random.default_rng(42)

    labels = random.integers(0, 3, 100)
    ptypes = np.ones(100)
    probabilities = random.normal(size=100) if probs else None
    mass = random.normal(size=100)

    components = gchop.models.Components(
        labels=labels,
        ptypes=ptypes,
        probabilities=probabilities,
        m=mass,
        lmap={},
    )

    assert len(components) == 100

    expected_repr = (
        "<Components length=100, labels=['0', '1', '2'], "
        f"probabilities={probs}, lmap=False>"
    )
    assert repr(components) == expected_repr


@pytest.mark.model
@pytest.mark.parametrize("probs", [True, False])
def test_Components_bad_len(probs):
    random = np.random.default_rng(42)

    labels = random.integers(0, 3, 100)
    ptypes = np.ones(99)
    mass = random.normal(size=95)
    probabilities = random.normal(size=98) if probs else None

    with pytest.raises(ValueError):
        gchop.models.Components(
            labels=labels,
            ptypes=ptypes,
            probabilities=probabilities,
            m=mass,
            lmap={},
        )


@pytest.mark.model
@pytest.mark.parametrize("probs", [True, False])
def test_Components_to_dataframe(probs):
    random = np.random.default_rng(42)

    labels = random.integers(0, 3, 100)
    ptypes = np.ones(100)
    mass = random.normal(size=100)
    probabilities = random.normal(size=100) if probs else None

    components = gchop.models.Components(
        labels=labels,
        ptypes=ptypes,
        probabilities=probabilities,
        m=mass,
        lmap={},
    )

    expected = pd.DataFrame(
        {
            "m": mass,
            "labels": labels,
            "ptypes": ptypes,
            "lmap": labels.astype(object),
        }
    )

    if probs:
        probs_df = pd.DataFrame({"probs_0": probabilities})
        expected = pd.concat([expected, probs_df], axis=1)

    pd.testing.assert_frame_equal(components.to_dataframe(), expected)


@pytest.mark.model
@pytest.mark.parametrize("probs", [True, False])
def test_Components_describe(probs):
    random = np.random.default_rng(42)

    labels = random.integers(0, 3, 100)
    ptypes = np.ones(100)
    mass = random.normal(loc=1000933.2, scale=252304.96, size=100)
    probabilities = random.uniform(size=(100, 3)) if probs else None

    components = gchop.models.Components(
        labels=labels,
        ptypes=ptypes,
        probabilities=probabilities,
        m=mass,
        lmap={},
    )

    expected_dict = {
        ("Particles", "Size"): {0: 27, 1: 33, 2: 40},
        ("Particles", "Fraction"): {0: 0.27, 1: 0.33, 2: 0.4},
        ("Deterministic mass", "Size"): {
            0: 26087288.44203574,
            1: 32763518.959382985,
            2: 38057037.01262353,
        },
        ("Deterministic mass", "Fraction"): {
            0: 0.26919687048838753,
            1: 0.33808944113336864,
            2: 0.39271368837824394,
        },
    }

    if probs:
        expected_dict.update(
            {
                ("Probabilistic mass", "Size"): {
                    0: 50689931.17716688,
                    1: 46151058.83650705,
                    2: 48717115.96184553,
                },
                ("Probabilistic mass", "Fraction"): {
                    0: 0.523073560078298,
                    1: 0.4762365638773781,
                    2: 0.5027159179570635,
                },
            }
        )

    expected = pd.DataFrame.from_dict(expected_dict)
    result = components.describe()

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


# =============================================================================
# DECOMPOSEDGALAXY
# =============================================================================


@pytest.mark.model
def test_Decomposedgalaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_stellar_attributes(self):
            return ["eps"]

        def split(self, X, y, attributes):
            return np.full(len(X), 100), None

        def get_rows_mask(self, X, y, attributes):
            return y == gchop.ParticleSetType.STARS.value

    decomposer = Decomposer()

    gal_decomp = decomposer.decompose(gal)
    gal_components = gal_decomp.components

    assert len(gal_decomp) == len(gal)

    expected_repr = repr(gal) + "\n" + repr(gal_components)
    assert repr(gal_decomp) == expected_repr
