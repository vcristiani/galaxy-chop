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

from galaxychop import core, models

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
# CIRCULARITY PLOTS
# =============================================================================


# get_circ_df_and_hue =========================================================
@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_labels_Component(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    circ = gal.stellar_dynamics()

    cps = models.DecomposedParticleSet(
        ptype=gal.stars.ptype,
        m=np.random.random(size=len(circ.eps)),
        x=gal.stars.x[: len(circ.eps)].copy(),
        y=gal.stars.y[: len(circ.eps)].copy(),
        z=gal.stars.z[: len(circ.eps)].copy(),
        vx=gal.stars.vx[: len(circ.eps)].copy(),
        vy=gal.stars.vy[: len(circ.eps)].copy(),
        vz=gal.stars.vz[: len(circ.eps)].copy(),
        potential=None,
        softening=float(gal.stars.softening.value),
        components=np.full(len(circ.eps), 100),
        labels=circ.eps,
        probabilities=np.zeros((len(circ.eps), 1)),
        has_probabilities=True,
    )

    df, hue = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels=cps,
        lmap=None,
    )

    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )
    # DecomposedParticleSet stores labels as strings; since the labels
    # are eps, each row's label must match that same row's eps column
    assert len(df) == mask.sum()
    expected = np.asarray(df["eps"]).astype(str)
    result = np.asarray(df[hue], dtype=str)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_labels_external_labels_list(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    circ = gal.stellar_dynamics()

    df, hue = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels=list(circ.eps_r),
        lmap=None,
    )

    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    assert (np.sort(df[hue]) == np.sort(circ.eps_r[mask])).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_labels_external_labels(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    circ = gal.stellar_dynamics()

    df, hue = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels=circ.eps_r,
        lmap=None,
    )

    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    assert (np.sort(df[hue]) == np.sort(circ.eps_r[mask])).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_labels_not_in_attributes(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, hue = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None,
        attributes=["normalized_star_energy", "eps"],
        labels="eps_r",
        lmap=None,
    )

    circ = gal.stellar_dynamics()
    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    assert (np.sort(df[hue]) == np.sort(circ.eps_r[mask])).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_labels_in_attributes(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, hue = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels="eps_r",
        lmap=None,
    )

    circ = gal.stellar_dynamics()
    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    assert (np.sort(df[hue]) == np.sort(circ.eps_r[mask])).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_lmap_map(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    circ = gal.stellar_dynamics()

    lmap = dict.fromkeys(circ.eps_r, 1)

    df, hue = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels="eps_r",
        lmap=lmap,
    )

    assert (df[hue] == 1).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_sdyn_df_and_hue_lmap_callable(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    def lmap(label):
        return 1

    df, hue = plotter.get_sdyn_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels="eps_r",
        lmap=lmap,
    )

    assert (df[hue] == 1).all()


# PLOTS =======================================================================


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametrize("img_format", ["png"])
def test_GalaxyPlotter_sdyn_pairplot(read_hdf5_galaxy, img_format):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_grid = plotter.sdyn_pairplot()

    # expected
    circ = gal.stellar_dynamics()
    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.normalized_star_Jz)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    df = pd.DataFrame(
        {
            "normalized_star_energy": circ.normalized_star_energy[mask],
            "normalized_star_Jz": circ.normalized_star_Jz[mask],
            "eps": circ.eps[mask],
            "eps_r": circ.eps_r[mask],
        }
    )
    expected_grid = sns.pairplot(df, kind="hist", diag_kind="kde")

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

    circ = gal.stellar_dynamics()
    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    df = pd.DataFrame({"eps": circ.eps[mask]})
    sns.histplot(x="eps", data=df, ax=exp_ax)
    exp_ax.set_xlabel("eps")


@pytest.mark.plot
@pytest.mark.slow
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_sdyn_kde(read_hdf5_galaxy, fig_test, fig_ref):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.sdyn_kde("eps", ax=test_ax)

    exp_ax = fig_ref.subplots()

    circ = gal.stellar_dynamics()
    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    df = pd.DataFrame({"eps": circ.eps[mask]})
    sns.kdeplot(x="eps", data=df, ax=exp_ax)
    exp_ax.set_xlabel("eps")
