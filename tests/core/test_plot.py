# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test plots"""

# =============================================================================
# IMPORTS
# =============================================================================

from unittest import mock

from galaxychop import core, models, preproc

from matplotlib.testing.decorators import (
    _image_directories,
    check_figures_equal,
    compare_images,
)

import numpy as np

import pandas as pd

import pytest

import seaborn as sns


# =============================================================================
# UTILITIES
# =============================================================================


def image_paths(func, format):
    idir = _image_directories(func)[-1]
    idir.mkdir(parents=True, exist_ok=True)

    test = idir / f"{func.__name__}[{format}]-.{format}"
    expected = idir / f"{func.__name__}[{format}]-expected.{format}"

    return test, expected


def assert_same_image(test_func, format, test_img, ref_img, **kwargs):
    # As the pairplot crap doesn't get neither axes nor figures I can't
    # use the check_figures_equals functions here, so we have to do
    # everything by hand...

    test_path, ref_path = image_paths(test_func, format)

    test_img.savefig(test_path, format=format)
    ref_img.savefig(ref_path, format=format)

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

    # not callable
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
def test_GalaxyPlotter_get_df_and_hue_labels_Components(galaxy):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    components = models.Components(
        labels=np.full(len(gal), 100),
        ptypes=np.full(len(gal), "foo"),
        m=np.random.random(size=len(gal)),
        probabilities=None,
        lmap={},
    )

    df, hue = plotter.get_df_and_hue(
        ptypes=None, attributes=None, labels=components, lmap=None
    )

    assert (df[hue] == components.labels).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_df_and_hue_labels_in_attributes(galaxy):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, hue = plotter.get_df_and_hue(
        ptypes=None, attributes=None, labels="x", lmap=None
    )

    expected = np.sort(gal.to_dataframe(attributes=["x"]).x)
    result = np.sort(df[hue])

    assert (result == expected).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_df_and_hue_lmap_map(galaxy):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    lmap = {"stars": "S", "dark_matter": "DM", "gas": "G"}

    df, hue = plotter.get_df_and_hue(
        ptypes=None, attributes=None, labels="ptype", lmap=lmap
    )

    ptype = gal.to_dataframe(attributes=["ptype"]).sort_values("ptype").ptype
    mapped = ptype.apply(lmap.get)

    assert (df[hue] == mapped).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_df_and_hue_lmap_callable(galaxy):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    lmap = {"stars": "S", "dark_matter": "DM", "gas": "G"}

    def lmap_func(label):
        return lmap.get(label, label)

    df, hue = plotter.get_df_and_hue(
        ptypes=None, attributes=None, labels="ptype", lmap=lmap_func
    )

    ptype = gal.to_dataframe(attributes=["ptype"]).sort_values("ptype").ptype
    mapped = ptype.apply(lmap_func)

    assert (df[hue] == mapped).all()


# PLOTS =======================================================================
@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("format", ["png"])
def test_GalaxyPlotter_pairplot(galaxy, format):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_grid = plotter.pairplot(attributes=["x", "y"])

    # EXPECTED
    df = gal.to_dataframe(attributes=["x", "y", "ptype"]).sort_values("ptype")
    expected_grid = sns.pairplot(
        data=df, hue="ptype", kind="hist", diag_kind="kde"
    )

    assert_same_image(
        test_GalaxyPlotter_pairplot, format, test_grid, expected_grid
    )


@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("format", ["png"])
def test_GalaxyPlotter_pairplot_external_labels(galaxy, format):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df = gal.to_dataframe(attributes=["x", "y", "ptype"])
    test_grid = plotter.pairplot(
        attributes=df[["x", "y"]], labels=df.ptype.to_numpy()
    )

    # EXPECTED
    df = gal.to_dataframe(attributes=["x", "y", "ptype"]).sort_values("ptype")
    df.columns = ["x", "y", "Labels"]
    expected_grid = sns.pairplot(
        data=df, hue="Labels", kind="hist", diag_kind="kde"
    )

    assert_same_image(
        test_GalaxyPlotter_pairplot_external_labels,
        format,
        test_grid,
        expected_grid,
    )


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_scatter(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.scatter("x", "y", labels="ptype", ptypes=["gas"], ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(
        ptypes=["gas"], attributes=["x", "y", "ptype"]
    ).sort_values("ptype")
    sns.scatterplot(data=df, x="x", y="y", hue="ptype", ax=exp_ax, marker=".")


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_hist(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.hist("x", y="y", labels="ptype", ptypes=["gas"], ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.histplot(data=df, x="x", y="y", hue="ptype", ax=exp_ax)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_kde(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.kde("x", y="y", labels="ptype", ptypes=["gas"], ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.kdeplot(data=df, x="x", y="y", hue="ptype", ax=exp_ax)


# =============================================================================
# CIRCULARITY PLOTS
# =============================================================================


# get_circ_df_and_hue =========================================================
@pytest.mark.plot
def test_GalaxyPlotter_get_circ_df_and_hue_labels_Component(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    circ = gal.stellar_dynamics()

    components = models.Components(
        labels=circ.eps,
        ptypes=np.full(len(circ.eps), "foo"),
        m=np.random.random(size=len(circ.eps)),
        probabilities=None,
        lmap={},
    )

    df, hue = plotter.get_circ_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels=components,
        lmap=None,
    )

    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )
    expected = np.sort(circ.eps[mask])
    result = np.sort(df[hue])
    assert (result == expected).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_circ_df_and_hue_labels_external_labels_list(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    circ = gal.stellar_dynamics()

    df, hue = plotter.get_circ_df_and_hue(
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
def test_GalaxyPlotter_get_circ_df_and_hue_labels_external_labels(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    circ = gal.stellar_dynamics()

    df, hue = plotter.get_circ_df_and_hue(
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
def test_GalaxyPlotter_get_circ_df_and_hue_labels_not_in_attributes(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, hue = plotter.get_circ_df_and_hue(
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
def test_GalaxyPlotter_get_circ_df_and_hue_labels_in_attributes(
    read_hdf5_galaxy,
):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    df, hue = plotter.get_circ_df_and_hue(
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
def test_GalaxyPlotter_get_circ_df_and_hue_lmap_map(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    circ = gal.stellar_dynamics()

    lmap = dict.fromkeys(circ.eps_r, 1)

    df, hue = plotter.get_circ_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels="eps_r",
        lmap=lmap,
    )

    assert (df[hue] == 1).all()


@pytest.mark.plot
def test_GalaxyPlotter_get_circ_df_and_hue_lmap_callable(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    def lmap(label):
        return 1

    df, hue = plotter.get_circ_df_and_hue(
        sdyn_kws=None,
        attributes=None,
        labels="eps_r",
        lmap=lmap,
    )

    assert (df[hue] == 1).all()


# PLOTS =======================================================================


@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("format", ["png"])
def test_GalaxyPlotter_circ_pairplot(read_hdf5_galaxy, format):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_grid = plotter.circ_pairplot()

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
        test_GalaxyPlotter_circ_pairplot,
        format,
        test_grid,
        expected_grid,
    )


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_circ_scatter(read_hdf5_galaxy, fig_test, fig_ref):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.circ_scatter("eps", "eps_r", ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    circ = gal.stellar_dynamics()
    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    df = pd.DataFrame({"eps": circ.eps[mask], "eps_r": circ.eps_r[mask]})
    sns.scatterplot(x="eps", y="eps_r", data=df, ax=exp_ax, marker=".")
    exp_ax.set_xlabel("eps")
    exp_ax.set_ylabel("eps_r")


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_circ_hist(read_hdf5_galaxy, fig_test, fig_ref):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.circ_hist("eps", ax=test_ax)

    # expected
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


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal(extensions=["png"])
def test_GalaxyPlotter_circ_kde(read_hdf5_galaxy, fig_test, fig_ref):
    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = core.plot.GalaxyPlotter(galaxy=gal)

    test_ax = fig_test.subplots()
    plotter.circ_kde("eps", ax=test_ax)

    # expected
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
