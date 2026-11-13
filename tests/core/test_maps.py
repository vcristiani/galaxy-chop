# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2020, 2021, 2022, 2023, 2024, 2025, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test projected maps"""

# =============================================================================
# IMPORTS
# =============================================================================

import galaxychop as gchop
from galaxychop import core

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# HELPERS
# =============================================================================


def _pset_kws(suffix, m, x, y, z, vx, vy, vz):
    return {
        f"m_{suffix}": np.asarray(m, dtype=float),
        f"x_{suffix}": np.asarray(x, dtype=float),
        f"y_{suffix}": np.asarray(y, dtype=float),
        f"z_{suffix}": np.asarray(z, dtype=float),
        f"vx_{suffix}": np.asarray(vx, dtype=float),
        f"vy_{suffix}": np.asarray(vy, dtype=float),
        f"vz_{suffix}": np.asarray(vz, dtype=float),
        f"softening_{suffix}": 0.0,
        f"potential_{suffix}": None,
    }


def _toy_galaxy(stars, dm=None, gas=None):
    # a far-away particle of each other type, so they can't sneak into
    # the stars maps
    far = dict(m=[1], x=[100], y=[100], z=[100], vx=[0], vy=[0], vz=[0])
    return gchop.mkgalaxy(
        **_pset_kws("s", **stars),
        **_pset_kws("dm", **(dm or far)),
        **_pset_kws("g", **(gas or far)),
    )


# =============================================================================
# TESTS
# =============================================================================


def test_projected_maps_edge_on_values():
    # two pixels per side, half width 2 -> pixel centers at -1 and 1
    gal = _toy_galaxy(
        stars=dict(
            m=[1, 3, 2],
            x=[-1.5, -0.5, 1.0],
            y=[9, 9, 9],  # line of sight: doesn't matter edge-on
            z=[-1.0, -1.0, 1.0],
            vx=[0, 0, 0],
            vy=[10, 30, -5],
            vz=[0, 0, 0],
        )
    )

    maps = gal.projected_maps(projection="edge-on", bins=2, half_width=2)

    assert list(maps.columns) == [
        "x",
        "z",
        "n_particles",
        "mass",
        "surface_density",
        "v_los",
        "sigma_los",
    ]
    # horizontal axis runs first
    np.testing.assert_array_equal(maps["x"], [-1, 1, -1, 1])
    np.testing.assert_array_equal(maps["z"], [-1, -1, 1, 1])
    np.testing.assert_array_equal(maps["n_particles"], [2, 0, 0, 1])
    np.testing.assert_allclose(maps["mass"], [4, 0, 0, 2])
    np.testing.assert_allclose(maps["surface_density"], [1, 0, 0, 0.5])

    # mass weighted: (1 * 10 + 3 * 30) / 4 = 25
    # sigma**2 = (1 * 15**2 + 3 * 5**2) / 4 = 75
    np.testing.assert_allclose(
        maps["v_los"], [25, np.nan, np.nan, -5], equal_nan=True
    )
    np.testing.assert_allclose(
        maps["sigma_los"], [np.sqrt(75), np.nan, np.nan, 0], equal_nan=True
    )

    assert maps.attrs == {
        "projection": "edge-on",
        "bins": 2,
        "half_width": 2.0,
        "pixel_size": 2.0,
        "min_particles": 1,
        "components": None,
    }


def test_projected_maps_face_on_uses_xy_and_vz():
    gal = _toy_galaxy(
        stars=dict(
            m=[1, 1],
            x=[-1, 1],
            y=[-1, 1],
            z=[50, -50],  # line of sight: doesn't matter face-on
            vx=[0, 0],
            vy=[0, 0],
            vz=[7, -7],
        )
    )

    maps = gal.projected_maps(projection="face-on", bins=2, half_width=2)

    assert list(maps.columns[:2]) == ["x", "y"]
    np.testing.assert_array_equal(maps["n_particles"], [1, 0, 0, 1])
    np.testing.assert_allclose(
        maps["v_los"], [7, np.nan, np.nan, -7], equal_nan=True
    )


def test_projected_maps_drops_outside_and_keeps_outer_edge():
    gal = _toy_galaxy(
        stars=dict(
            m=[1, 1, 1],
            x=[2.0, 2.5, 0.0],  # on the edge, outside, inside
            y=[0, 0, 0],
            z=[2.0, 0.0, -3.0],  # on the edge, inside, outside
            vx=[0, 0, 0],
            vy=[0, 0, 0],
            vz=[0, 0, 0],
        )
    )

    maps = gal.projected_maps(bins=2, half_width=2)

    np.testing.assert_array_equal(maps["n_particles"], [0, 0, 0, 1])


def test_projected_maps_min_particles():
    gal = _toy_galaxy(
        stars=dict(
            m=[1, 1, 1],
            x=[-1, -1, 1],
            y=[0, 0, 0],
            z=[-1, -1, 1],
            vx=[0, 0, 0],
            vy=[1, 3, 5],
            vz=[0, 0, 0],
        )
    )

    maps = gal.projected_maps(bins=2, half_width=2, min_particles=2)

    # the lone particle keeps its mass but loses its velocities
    np.testing.assert_array_equal(maps["n_particles"], [2, 0, 0, 1])
    np.testing.assert_allclose(maps["mass"], [2, 0, 0, 1])
    np.testing.assert_allclose(
        maps["v_los"], [2, np.nan, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        maps["sigma_los"], [1, np.nan, np.nan, np.nan], equal_nan=True
    )


def test_projected_maps_ptypes():
    stars = dict(m=[1], x=[-1], y=[0], z=[-1], vx=[0], vy=[0], vz=[0])
    gas = dict(m=[2], x=[1], y=[0], z=[1], vx=[0], vy=[0], vz=[0])
    gal = _toy_galaxy(stars=stars, gas=gas)

    only_gas = gal.projected_maps(ptypes=("gas",), bins=2, half_width=2)
    np.testing.assert_allclose(only_gas["mass"], [0, 0, 0, 2])

    pooled = gal.projected_maps(ptypes=("stars", "gas"), bins=2, half_width=2)
    np.testing.assert_allclose(pooled["mass"], [1, 0, 0, 2])


def test_projected_maps_default_half_width(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    stars = gal.stars.to_dataframe(attributes=["m", "x", "z"])

    maps = gal.projected_maps(bins=10)

    expected = np.percentile(
        np.hypot(stars["x"], stars["z"]),
        core.maps.DEFAULT_HALF_WIDTH_PCT,
    )
    assert maps.attrs["half_width"] == pytest.approx(expected)
    assert len(maps) == 100

    # every particle inside the map lands in exactly one pixel
    inside = (stars["x"].abs() <= expected) & (stars["z"].abs() <= expected)
    assert maps["n_particles"].sum() == inside.sum()
    assert maps["mass"].sum() == pytest.approx(stars["m"][inside].sum())


def test_Galaxy_projected_maps(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    from_gal = gal.projected_maps(projection="face-on", bins=8)
    from_func = core.maps.projected_maps(gal, projection="face-on", bins=8)

    pd.testing.assert_frame_equal(from_gal, from_func)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"projection": "side-on"}, "projection"),
        ({"bins": 0}, "bins"),
        ({"bins": 2.5}, "bins"),
        ({"half_width": 0}, "half_width"),
        ({"min_particles": 0}, "min_particles"),
    ],
)
def test_projected_maps_invalid(galaxy, kwargs, match):
    gal = galaxy(seed=42)
    with pytest.raises(ValueError, match=match):
        gal.projected_maps(**kwargs)


# =============================================================================
# DECOMPOSED GALAXY
# =============================================================================


@pytest.fixture(scope="module")
def decomposed_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.star_align(gchop.preproc.center(gal))
    return gchop.decomposers.JThreshold().decompose(gal)


@pytest.mark.model
def test_DecomposedGalaxy_projected_maps_components_add_up(
    decomposed_galaxy,
):
    whole = decomposed_galaxy.projected_maps(bins=20)
    parts = [
        decomposed_galaxy.projected_maps(bins=20, components=label)
        for label in ("Disk", "Spheroid", "stars")
    ]

    # every component shares the grid of the whole galaxy...
    for part in parts:
        assert part.attrs["half_width"] == whole.attrs["half_width"]
        pd.testing.assert_frame_equal(part[["x", "z"]], whole[["x", "z"]])

    # ...so their pixels add up to it
    np.testing.assert_array_equal(
        sum(part["n_particles"] for part in parts), whole["n_particles"]
    )
    np.testing.assert_allclose(
        sum(part["mass"] for part in parts), whole["mass"]
    )


@pytest.mark.model
def test_DecomposedGalaxy_projected_maps_pooled_components(
    decomposed_galaxy,
):
    disk = decomposed_galaxy.projected_maps(bins=20, components="Disk")
    pooled = decomposed_galaxy.projected_maps(
        bins=20, components=("Disk", "Spheroid")
    )
    spheroid = decomposed_galaxy.projected_maps(bins=20, components="Spheroid")

    assert disk.attrs["components"] == ("Disk",)
    assert pooled.attrs["components"] == ("Disk", "Spheroid")
    np.testing.assert_allclose(pooled["mass"], disk["mass"] + spheroid["mass"])


@pytest.mark.model
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"components": "Bulge"}, "Unknown components"),
        # gas particles are labeled "gas", there are no gas disk particles
        ({"components": "Disk", "ptypes": ("gas",)}, "No particles"),
        ({"components": ()}, "No particles"),
    ],
)
def test_DecomposedGalaxy_projected_maps_invalid(
    decomposed_galaxy, kwargs, match
):
    with pytest.raises(ValueError, match=match):
        decomposed_galaxy.projected_maps(bins=4, **kwargs)


def test_projected_maps_components_needs_decomposed_galaxy(galaxy):
    gal = galaxy(seed=42)
    with pytest.raises(TypeError, match="decomposed galaxy"):
        core.maps.projected_maps(gal, components="Disk")
