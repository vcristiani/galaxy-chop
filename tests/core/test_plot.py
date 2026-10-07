# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test plots"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from unittest import mock

from astropy import units as u

from galaxychop import core

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pandas as pd

import pytest

import seaborn as sns

# Incompatible con Python 3.9
# no vi otra forma de arreglarlo que no sea saltando
# los tests en 3.9
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Seaborn 0.11.x plotting is incompatible with Python 3.9",
)
# =============================================================================
# TEST __call__
# =============================================================================


@pytest.mark.plot
@pytest.mark.parametrize(
    "pkind", core.plot.GalaxyPlotter._P_KIND_FORBIDEN_METHODS
)
def test_GalaxyPlotter_call_invalid_forbiden_plot_kind(galaxy, pkind):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    with pytest.raises(ValueError):
        plotter(pkind)


@pytest.mark.plot
def test_GalaxyPlotter_call_invalid_plot_kind(galaxy):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    with pytest.raises(ValueError):
        plotter("__call__")

    super(core.plot.GalaxyPlotter, plotter).__setattr__("zaraza", None)
    with pytest.raises(ValueError):
        plotter("zaraza")


@pytest.mark.plot
@pytest.mark.parametrize("plot_kind", ["hist"])
def test_GalaxyPlotter_call(galaxy, plot_kind):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    method_name = f"galaxychop.core.plot.GalaxyPlotter.{plot_kind}"

    with mock.patch(method_name) as plot_method:
        plotter(plot_kind=plot_kind)

    plot_method.assert_called_once()


# =============================================================================
# REAL SPACE PLOT
# =============================================================================

# get_df_and_hue ==============================================================


@pytest.mark.plot
def test_GalaxyPlotter_get_df_and_hue_lmap_map(galaxy):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    lmap = {"stars": "estrella", "dark_matter": "materia", "gas": "gas"}

    df, style = plotter.get_df_and_hue(ptypes=None, attributes=None, lmap=lmap)

    base = gal.to_dataframe(attributes=["ptype"]).ptype.map(lmap)
    assert (df["ptype"].astype(str).to_numpy() == base.to_numpy()).all()
    assert set(style["hue_order"]) == set(lmap.values())
    assert style["palette"]["estrella"] == "tab:red"
    assert style["palette"]["gas"] == "tab:blue"
    assert style["linestyles"]["gas"] == "--"


@pytest.mark.plot
def test_GalaxyPlotter_get_df_and_hue_lmap_callable(galaxy):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    def lmap_func(label):
        return label.upper()

    df, style = plotter.get_df_and_hue(
        ptypes=None, attributes=None, lmap=lmap_func
    )

    base = gal.to_dataframe(attributes=["ptype"]).ptype.map(lmap_func)
    assert (df["ptype"].astype(str).to_numpy() == base.to_numpy()).all()
    assert set(style["hue_order"]) == {"STARS", "GAS", "DARK_MATTER"}


# Same fixed look the plotter uses for each particle type
PTYPE_PALETTE = {
    "stars": "tab:red",
    "gas": "tab:blue",
    "dark_matter": "#222222",
}


def _zoom(ax, df, x, y=None, pct=1, margin=0.1):
    """Match GalaxyPlotter._zoom_to_data for the hand-built reference axes."""
    lo, hi = np.percentile(df[x], [pct, 100 - pct])
    pad = (hi - lo) * margin
    ax.set_xlim(lo - pad, hi + pad)
    if y is not None:
        lo, hi = np.percentile(df[y], [pct, 100 - pct])
        pad = (hi - lo) * margin
        ax.set_ylim(lo - pad, hi + pad)


# PLOTS =======================================================================
@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_hist2d(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.hist2d("x", y="y", ptypes=["gas"], ax=test_ax)

    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.histplot(
        data=df,
        x="x",
        y="y",
        hue="ptype",
        hue_order=["gas"],
        palette=PTYPE_PALETTE,
        ax=exp_ax,
    )
    kpc = u.kpc.to_string("latex")
    exp_ax.set_xlabel(f"x [{kpc}]")
    exp_ax.set_ylabel(f"y [{kpc}]")
    _zoom(exp_ax, df, "x", "y")
    exp_ax.set_box_aspect(1)


@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_hist(galaxy, fig_test, fig_ref):
    """hist(x) is a shortcut for hist2d(x, y=None)."""
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.hist("x", ax=test_ax)

    exp_ax = fig_ref.subplots()
    plotter.hist2d("x", y=None, ax=exp_ax)


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_kde2d(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.kde2d("x", y="y", ptypes=["gas"], ax=test_ax)

    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.kdeplot(
        data=df,
        x="x",
        y="y",
        fill=False,
        color=PTYPE_PALETTE["gas"],
        linestyles="--",
        ax=exp_ax,
    )
    kpc = u.kpc.to_string("latex")
    exp_ax.set_xlabel(f"x [{kpc}]")
    exp_ax.set_ylabel(f"y [{kpc}]")
    _zoom(exp_ax, df, "x", "y")
    exp_ax.set_box_aspect(1)


@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_kde(galaxy, fig_test, fig_ref):
    """kde(x) is a shortcut for kde2d(x, y=None)."""
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.kde("x", ax=test_ax)

    exp_ax = fig_ref.subplots()
    plotter.kde2d("x", y=None, ax=exp_ax)


@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_rotation_curve(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.rotation_curve(ptypes=["gas"], ax=test_ax)

    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(
        ptypes=["gas"],
        attributes=["radius", "circular_velocity"],
        galaxy_circular_velocity=True,
    ).sort_values("radius")
    # whole galaxy: a single black curve
    sns.lineplot(
        data=df,
        x="radius",
        y="galaxy_circular_velocity",
        estimator=None,
        color="black",
        linestyle="-",
        label="galaxy",
        ax=exp_ax,
    )
    # gas on its own, self-contained circular velocity
    sns.lineplot(
        data=df,
        x="radius",
        y="circular_velocity",
        estimator=None,
        color=PTYPE_PALETTE["gas"],
        linestyle="--",
        label="gas",
        ax=exp_ax,
    )
    exp_ax.set_xlabel(f"radius [{u.kpc.to_string('latex')}]")
    exp_ax.set_ylabel(f"circular velocity [{(u.km / u.s).to_string('latex')}]")
    exp_ax.legend()


@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_rotation_curve_no_galaxy(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.rotation_curve(ptypes=["gas"], galaxy=False, ax=test_ax)

    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(
        ptypes=["gas"],
        attributes=["radius", "circular_velocity"],
        galaxy_circular_velocity=False,
    ).sort_values("radius")
    # no whole-galaxy curve this time: only gas, self-contained
    sns.lineplot(
        data=df,
        x="radius",
        y="circular_velocity",
        estimator=None,
        color=PTYPE_PALETTE["gas"],
        linestyle="--",
        label="gas",
        ax=exp_ax,
    )
    exp_ax.set_xlabel(f"radius [{u.kpc.to_string('latex')}]")
    exp_ax.set_ylabel(f"circular velocity [{(u.km / u.s).to_string('latex')}]")
    exp_ax.legend()


# =============================================================================
# STELLAR DYNAMICS PLOTS
# =============================================================================


def _sdyn_finite_df(gal, attributes):
    """Expected stellar dynamics dataframe: finite rows of the attributes."""
    circ = gal.stellar_dynamics()
    mask = circ.isfinite()
    return pd.DataFrame(
        {aname: getattr(circ, aname)[mask] for aname in attributes}
    )


# get_sdyn_df_and_hue =========================================================
@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, style = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None, attributes=["eps", "eps_r"], lmap=None
    )

    expected = _sdyn_finite_df(gal, ["eps", "eps_r"])
    assert list(df.columns) == ["eps", "eps_r", "ptype"]
    np.testing.assert_array_equal(df["eps"], expected["eps"])
    np.testing.assert_array_equal(df["eps_r"], expected["eps_r"])
    assert (df["ptype"] == "stars").all()
    assert style == {
        "hue_order": ["stars"],
        "palette": {"stars": "tab:red"},
        "linestyles": {"stars": "-"},
    }


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_all_attributes(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, _ = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None, attributes=None, lmap=None
    )

    sdyn_keys = list(gal.stellar_dynamics().to_dict())
    assert list(df.columns) == sdyn_keys + ["ptype"]


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_lmap_map(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, style = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None, attributes=["eps"], lmap={"stars": "estrella"}
    )

    assert (df["ptype"] == "estrella").all()
    assert style["hue_order"] == ["estrella"]
    assert style["palette"] == {"estrella": "tab:red"}


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_lmap_callable(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, style = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None, attributes=["eps"], lmap=str.upper
    )

    assert (df["ptype"] == "STARS").all()
    assert style["hue_order"] == ["STARS"]


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_invalid_lmap(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    with pytest.raises(TypeError):
        plotter.get_sdyn_df_and_hue(sdyn_kws=None, attributes=None, lmap=1)


# PLOTS =======================================================================
@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_sdyn_hist2d(read_hdf5_galaxy, fig_test, fig_ref):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.sdyn_hist2d("eps", y=None, ax=test_ax)

    exp_ax = fig_ref.subplots()

    df = _sdyn_finite_df(gal, ["eps"])
    df["ptype"] = "stars"
    sns.histplot(
        x="eps",
        data=df,
        hue="ptype",
        hue_order=["stars"],
        palette=PTYPE_PALETTE,
        ax=exp_ax,
    )
    _zoom(exp_ax, df, "eps")
    exp_ax.set_box_aspect(1)


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_sdyn_hist(read_hdf5_galaxy, fig_test, fig_ref):
    """sdyn_hist(x) is a shortcut for sdyn_hist2d(x, y=None)."""
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.sdyn_hist("eps", ax=test_ax)

    exp_ax = fig_ref.subplots()
    plotter.sdyn_hist2d("eps", y=None, ax=exp_ax)


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_sdyn_kde2d(read_hdf5_galaxy, fig_test, fig_ref):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.sdyn_kde2d("eps", y=None, ax=test_ax)

    exp_ax = fig_ref.subplots()

    df = _sdyn_finite_df(gal, ["eps"])
    sns.kdeplot(
        x="eps",
        data=df,
        fill=False,
        color=PTYPE_PALETTE["stars"],
        linestyle="-",
        ax=exp_ax,
    )
    _zoom(exp_ax, df, "eps")
    exp_ax.set_box_aspect(1)


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_sdyn_kde(read_hdf5_galaxy, fig_test, fig_ref):
    """sdyn_kde(x) is a shortcut for sdyn_kde2d(x, y=None)."""
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.sdyn_kde("eps", ax=test_ax)

    exp_ax = fig_ref.subplots()
    plotter.sdyn_kde2d("eps", y=None, ax=exp_ax)
