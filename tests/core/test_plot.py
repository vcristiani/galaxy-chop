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

from galaxychop import core

from matplotlib.testing.decorators import (
    _image_directories,
    check_figures_equal,
    compare_images,
)

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
# UTILITIES
# =============================================================================


def image_paths(func, img_format):
    idir = _image_directories(func)[-1]
    idir.mkdir(parents=True, exist_ok=True)

    test = idir / f"{func.__name__}[{img_format}]-.{img_format}"
    expected = idir / f"{func.__name__}[{img_format}]-expected.{img_format}"

    return test, expected


def assert_same_image(test_func, img_format, test_img, ref_img, **kwargs):
    test_path, ref_path = image_paths(test_func, img_format)

    test_img.savefig(test_path, format=img_format)
    ref_img.savefig(ref_path, format=img_format)

    kwargs.setdefault("tol", 0)
    result = compare_images(test_path, ref_path, **kwargs)

    if result:
        pytest.fail(result)


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
@pytest.mark.parametrize("plot_kind", ["pairplot"])
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
    assert style["palette"]["estrella"] == "black"
    assert style["palette"]["gas"] == "#7f7f7f"
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
PTYPE_PALETTE = {"stars": "black", "gas": "#7f7f7f", "dark_matter": "#bdbdbd"}


# PLOTS =======================================================================
@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("img_format", ["png"])
def test_GalaxyPlotter_pairplot(galaxy, img_format):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_grid = plotter.pairplot(attributes=["x", "y"])

    # EXPECTED
    df = gal.to_dataframe(attributes=["x", "y", "ptype"])
    expected_grid = sns.pairplot(
        data=df,
        hue="ptype",
        hue_order=["dark_matter", "gas", "stars"],
        palette=PTYPE_PALETTE,
        kind="hist",
        diag_kind="kde",
        diag_kws={"fill": False},
    )

    assert_same_image(
        test_GalaxyPlotter_pairplot, img_format, test_grid, expected_grid
    )


@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_hist(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.hist("x", y="y", ptypes=["gas"], ax=test_ax)

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


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_kde(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.kde("x", y="y", ptypes=["gas"], ax=test_ax)

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
        "palette": {"stars": "black"},
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
    assert style["palette"] == {"estrella": "black"}


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
@pytest.mark.parametrize("img_format", ["png"])
def test_GalaxyPlotter_sdyn_pairplot(read_hdf5_galaxy, img_format):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_grid = plotter.sdyn_pairplot()

    # expected
    df = _sdyn_finite_df(
        gal, ["normalized_star_energy", "normalized_star_Jz", "eps", "eps_r"]
    )
    df["ptype"] = "stars"
    expected_grid = sns.pairplot(
        df,
        hue="ptype",
        hue_order=["stars"],
        palette=PTYPE_PALETTE,
        kind="hist",
        diag_kind="kde",
        diag_kws={"fill": False},
    )

    assert_same_image(
        test_GalaxyPlotter_sdyn_pairplot,
        img_format,
        test_grid,
        expected_grid,
    )


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_sdyn_hist(read_hdf5_galaxy, fig_test, fig_ref):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.sdyn_hist("eps", y=None, ax=test_ax)

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


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_sdyn_kde(read_hdf5_galaxy, fig_test, fig_ref):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.sdyn_kde("eps", y=None, ax=test_ax)

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
