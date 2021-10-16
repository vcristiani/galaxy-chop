import warnings

import numpy as np

import galaxychop as gchop


def test_GalaxyDecomposerABC_attributes_repr():
    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            return ["normalized_star_energy", "eps", "eps_r"]

        def get_ptypes(self):
            return ["stars"]

        def split(self):
            ...

        def valid_rows(self, X, t, attributes):
            ...

    decomposer = Decomposer()
    result = repr(decomposer)
    expected = (
        "Decomposer(bins=(0.05, 0.005), "
        "ptypes=['stars'], "
        "attributes=['normalized_star_energy', 'eps', 'eps_r'])"
    )

    assert result == expected


def test_GalaxyDecomposerABC_attributes_matrix(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.star_align(gchop.center(gal))

    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_attributes(self):
            ...

        def get_ptypes(self):
            ...

        def split(self):
            ...

        def valid_rows(self, X, t, attributes):
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
