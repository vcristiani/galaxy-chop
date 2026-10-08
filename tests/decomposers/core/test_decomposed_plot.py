# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test plots for decomposed galaxies."""

# =============================================================================
# IMPORTS
# =============================================================================

import sys

import galaxychop as gchop
from galaxychop.core.galaxy import ParticleSetType
from galaxychop.decomposers.core.decomposed_galaxy import (
    DecomposedGalaxy,
    DecomposedParticleSet,
)
from galaxychop.decomposers.core.decomposed_plot import (
    DecomposedGalaxyPlotter,
    _component_key,
)

import matplotlib.pyplot as plt

import numpy as np

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Seaborn 0.11.x plotting is incompatible with Python 3.9",
)


# =============================================================================
# HELPERS
# =============================================================================


def _mk_decomposed_galaxy():
    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1.0, 2.0, 3.0, 4.0]),
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        z=np.array([2.0, 3.0, 4.0, 5.0]),
        vx=np.array([3.0, 4.0, 5.0, 6.0]),
        vy=np.array([4.0, 5.0, 6.0, 7.0]),
        vz=np.array([5.0, 6.0, 7.0, 8.0]),
        potential=np.array([-6.0, -7.0, -8.0, -9.0]),
        softening=0.1,
        components=np.array([0, 1, 0, 1]),
        labels=np.array(["disk", "halo", "disk", "halo"]),
        probabilities=np.full((4, 2), np.nan),
        has_probabilities=False,
    )
    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=np.array([5.0, 5.0]),
        x=np.array([0.0, 1.0]),
        y=np.array([1.0, 2.0]),
        z=np.array([2.0, 3.0]),
        vx=np.array([3.0, 4.0]),
        vy=np.array([4.0, 5.0]),
        vz=np.array([5.0, 6.0]),
        potential=np.array([-6.0, -7.0]),
        softening=0.1,
        components=np.array([0, 0]),
        labels=np.array(["dark_matter", "dark_matter"]),
        probabilities=np.full((2, 1), np.nan),
        has_probabilities=False,
    )
    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=np.array([2.0]),
        x=np.array([0.0]),
        y=np.array([1.0]),
        z=np.array([2.0]),
        vx=np.array([3.0]),
        vy=np.array([4.0]),
        vz=np.array([5.0]),
        potential=np.array([-6.0]),
        softening=0.1,
        components=np.array([0]),
        labels=np.array(["gas"]),
        probabilities=np.full((1, 1), np.nan),
        has_probabilities=False,
    )
    return DecomposedGalaxy(
        method="test",
        component_name_mapping={0: "disk", 1: "halo"},
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.model
def test_component_key():
    assert _component_key("Disk") == "disk"
    assert _component_key("Cold Disk") == "cold_disk"
    assert _component_key("dark_matter") == "dark_matter"
    assert _component_key("totally-unknown-xyz") == "no_component"
    assert _component_key(0.0) == "no_component"


@pytest.mark.model
def test_DecomposedGalaxy_plot_accessor():
    dgal = _mk_decomposed_galaxy()
    assert isinstance(dgal.plot, DecomposedGalaxyPlotter)
    assert dgal.plot is dgal.plot  # cached, same instance every access


@pytest.mark.plot
def test_DecomposedGalaxyPlotter_get_df_and_hue():
    dgal = _mk_decomposed_galaxy()

    df, style = dgal.plot.get_df_and_hue(
        ptypes=None, attributes=["x", "y"], lmap=None
    )

    expected = {"dark_matter", "gas", "disk", "halo"}
    assert set(df["ptype"].unique()) == expected
    assert set(style["hue_order"]) == expected
    for key in style["hue_order"]:
        assert key in style["palette"]
        assert key in style["linestyles"]
        assert key in style["alphas"]
        assert key in style["linewidths"]

    # dark_matter/gas keep the fixed look GalaxyPlotter always used
    assert style["palette"]["dark_matter"] == "#222222"
    assert style["palette"]["gas"] == "tab:blue"


@pytest.mark.plot
def test_DecomposedGalaxyPlotter_get_df_and_hue_ptypes_filter():
    dgal = _mk_decomposed_galaxy()

    df, style = dgal.plot.get_df_and_hue(
        ptypes=["stars"], attributes=["x", "y"], lmap=None
    )

    assert set(df["ptype"].unique()) == {"disk", "halo"}
    assert set(style["hue_order"]) == {"disk", "halo"}


@pytest.mark.model
def test_DecomposedGalaxyPlotter_get_df_and_hue_self_contained_vcirc():
    """Each component's circular_velocity is its own, not the whole type's.

    Otherwise every component of the same particle type would just
    replay the type's pooled ParticleSet.circular_velocity_ curve and
    overlap each other in rotation_curve().
    """
    dgal = _mk_decomposed_galaxy()

    df, _ = dgal.plot.get_df_and_hue(
        ptypes=["stars"],
        attributes=["radius", "circular_velocity"],
        lmap=None,
    )

    disk_vcirc = df.loc[df["ptype"] == "disk", "circular_velocity"]
    halo_vcirc = df.loc[df["ptype"] == "halo", "circular_velocity"]

    # neither matches the pooled, whole-stars curve ParticleSet exposes
    pooled_vcirc = dgal.stars.to_dataframe(attributes=["circular_velocity"])[
        "circular_velocity"
    ]
    assert not np.allclose(
        disk_vcirc.sort_index(), pooled_vcirc.loc[disk_vcirc.index]
    )
    assert not np.allclose(
        halo_vcirc.sort_index(), pooled_vcirc.loc[halo_vcirc.index]
    )
    # and they differ from each other, since disk/halo have different
    # mass-vs-radius distributions
    assert not np.allclose(
        sorted(disk_vcirc), sorted(halo_vcirc), equal_nan=True
    )


@pytest.mark.plot
@pytest.mark.slow
def test_DecomposedGalaxyPlotter_plots_smoke(read_hdf5_galaxy):
    """Every plot method should run and hue by component, not ptype.

    JHistogram only splits stars into disk/spheroid/unassigned, so dark
    matter and gas (each their own, single component) are expected to
    show up in the legend too, alongside those three.
    """
    gal = read_hdf5_galaxy("gal394242.h5")
    dgal = gchop.decomposers.JHistogram().decompose(gal)

    df, style = dgal.plot.get_sdyn_df_and_hue(
        sdyn_kws=None, attributes=["eps"], lmap=None
    )
    assert set(df["ptype"].unique()) == set(style["hue_order"])
    assert len(style["hue_order"]) > 1  # JHistogram splits stars

    for _, ax in (
        ("hist2d", dgal.plot.hist2d("x", y="y", ax=plt.figure().subplots())),
        ("kde2d", dgal.plot.kde2d("x", y="y", ax=plt.figure().subplots())),
        (
            "rotation_curve",
            dgal.plot.rotation_curve(galaxy=False, ax=plt.figure().subplots()),
        ),
        (
            "sdyn_hist2d",
            dgal.plot.sdyn_hist2d("eps", y=None, ax=plt.figure().subplots()),
        ),
        (
            "sdyn_kde2d",
            dgal.plot.sdyn_kde2d("eps", y=None, ax=plt.figure().subplots()),
        ),
        (
            "sdyn_kde2d-biv",
            dgal.plot.sdyn_kde2d("eps", ax=plt.figure().subplots()),
        ),
    ):
        legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
        assert legend_labels
        assert ax.get_legend().get_title().get_text() == ""
