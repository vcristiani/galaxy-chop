# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test plots"""

# =============================================================================
# IMPORTS
# =============================================================================

from unittest import mock

from galaxychop import plot, utils

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
# TEST get_df_and_hue
# =============================================================================


@pytest.mark.plot
def test_GalaxyPlotter_get_df_and_hue_lmap(galaxy):
    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    lmap = {"stars": "S", "dark_matter": "DM", "gas": "G"}

    df, hue = plotter.get_df_and_hue(
        ptypes=None, attributes=None, labels="ptype", lmap=lmap
    )

    ptype = gal.to_dataframe(attributes=["ptype"]).ptype
    mapped = ptype.apply(lmap.get)

    assert (df[hue] == mapped).all()


# =============================================================================
# TEST __call__
# =============================================================================


@pytest.mark.plot
@pytest.mark.parametrize("pkind", plot.GalaxyPlotter._P_KIND_FORBIDEN_METHODS)
def test_GalaxyPlotter_call_invalid_forbiden_plot_kind(galaxy, pkind):
    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    with pytest.raises(ValueError):
        plotter(pkind)


@pytest.mark.plot
def test_GalaxyPlotter_call_invalid_plot_kind(galaxy):
    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    with pytest.raises(ValueError):
        plotter("__call__")

    # not callable
    super(plot.GalaxyPlotter, plotter).__setattr__("zaraza", None)
    with pytest.raises(ValueError):
        plotter("zaraza")


@pytest.mark.plot
@pytest.mark.parametrize("plot_kind", ["pairplot"])
def test_GalaxyPlotter_call(galaxy, plot_kind):

    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    method_name = f"galaxychop.plot.GalaxyPlotter.{plot_kind}"

    with mock.patch(method_name) as plot_method:
        plotter(plot_kind=plot_kind)

    plot_method.assert_called_once()


# =============================================================================
# pairplot
# =============================================================================


@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("format", ["png", "pdf", "svg"])
def test_GalaxyPlotter_pairplot(galaxy, format):

    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)
    test_grid = plotter.pairplot(attributes=["x", "y"])

    # EXPECTED
    df = gal.to_dataframe(attributes=["x", "y", "ptype"])
    expected_grid = sns.pairplot(
        data=df, hue="ptype", kind="hist", diag_kind="kde"
    )

    assert_same_image(
        test_GalaxyPlotter_pairplot, format, test_grid, expected_grid
    )


@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("format", ["png", "pdf", "svg"])
def test_GalaxyPlotter_pairplot_external_labels(galaxy, format):

    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    df = gal.to_dataframe(attributes=["x", "y", "ptype"])
    test_grid = plotter.pairplot(
        attributes=df[["x", "y"]], labels=df.ptype.to_numpy()
    )

    # EXPECTED
    df = gal.to_dataframe(attributes=["x", "y", "ptype"])
    df.columns = ["x", "y", "Hue"]
    expected_grid = sns.pairplot(
        data=df, hue="Hue", kind="hist", diag_kind="kde"
    )

    assert_same_image(
        test_GalaxyPlotter_pairplot_external_labels,
        format,
        test_grid,
        expected_grid,
    )


@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("format", ["png", "pdf", "svg"])
def test_GalaxyPlotter_dis(galaxy, format):
    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)
    test_grid = plotter.dis("x", "y", labels="ptype", ptypes=["gas"])

    # EXPECTED
    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    expected_grid = sns.displot(x="x", y="y", data=df, hue="ptype")

    assert_same_image(test_GalaxyPlotter_dis, format, test_grid, expected_grid)


# =============================================================================
#
# =============================================================================


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
def test_GalaxyPlotter_scatter(galaxy, fig_test, fig_ref):

    gal = galaxy(seed=42)

    test_ax = fig_test.subplots()
    gal.plot.scatter("x", "y", labels="ptype", ptypes=["gas"], ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.scatterplot(data=df, x="x", y="y", hue="ptype", ax=exp_ax)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
def test_GalaxyPlotter_hist(galaxy, fig_test, fig_ref):

    gal = galaxy(seed=42)

    test_ax = fig_test.subplots()
    gal.plot.hist("x", "y", labels="ptype", ptypes=["gas"], ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.histplot(data=df, x="x", y="y", hue="ptype", ax=exp_ax)


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
def test_GalaxyPlotter_kde(galaxy, fig_test, fig_ref):

    gal = galaxy(seed=42)

    test_ax = fig_test.subplots()
    gal.plot.kde("x", "y", labels="ptype", ptypes=["gas"], ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.kdeplot(data=df, x="x", y="y", hue="ptype", ax=exp_ax)


# =============================================================================
# CIRCULARITY
# =============================================================================


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
def test_GalaxyPlotter_circ_hist(read_hdf5_galaxy, fig_test, fig_ref):

    gal = read_hdf5_galaxy("gal394242.h5")

    test_ax = fig_test.subplots()
    gal.plot.circ_hist(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    circ = utils.jcirc(gal)
    sns.histplot(circ.eps, ax=exp_ax)
    exp_ax.set_xlabel(r"$\epsilon$")


@pytest.mark.slow
@pytest.mark.plot
@check_figures_equal()
def test_GalaxyPlotter_circ_kde(read_hdf5_galaxy, fig_test, fig_ref):

    gal = read_hdf5_galaxy("gal394242.h5")

    test_ax = fig_test.subplots()
    gal.plot.circ_kde(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    circ = utils.jcirc(gal)
    sns.kdeplot(circ.eps, ax=exp_ax)
    exp_ax.set_xlabel(r"$\epsilon$")


@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("format", ["png"])
def test_GalaxyPlotter_circ_pairplot(read_hdf5_galaxy, format):

    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = plot.GalaxyPlotter(galaxy=gal)

    test_grid = plotter.circ_pairplot()

    # expected
    circ = utils.jcirc(gal)
    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    df = pd.DataFrame(
        {
            "Normalized star energy": circ.normalized_star_energy[mask],
            r"$\epsilon$": circ.eps[mask],
            r"$\epsilon_r$": circ.eps_r[mask],
        }
    )
    expected_grid = sns.pairplot(df, kind="hist", diag_kind="kde")

    assert_same_image(
        test_GalaxyPlotter_circ_pairplot,
        format,
        test_grid,
        expected_grid,
    )


@pytest.mark.xfail
@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("format", ["png"])
def test_GalaxyPlotter_circularity_component_labels(read_hdf5_galaxy, format):

    gal = read_hdf5_galaxy("gal394242.h5")
    plotter = plot.GalaxyPlotter(galaxy=gal)

    circ = utils.jcirc(gal)
    # decomposer = models.KMeans(n_clusters=2)
    # labels = decomposer.decompose(gal)
    # labels = labels.labels[labels.ptypes == 'stars']

    # expected
    mask = (
        np.isfinite(circ.normalized_star_energy)
        & np.isfinite(circ.eps)
        & np.isfinite(circ.eps_r)
    )

    labels1 = np.full((len(gal) - len(circ.eps)), np.nan)
    labels2 = np.ones(len(circ.eps_r[mask]))
    labels = np.hstack((labels1, labels2))

    # import ipdb; ipdb.set_trace()

    test_grid = plotter.circularity_components(labels=labels)

    df = pd.DataFrame(
        {
            "Normalized star energy": circ.normalized_star_energy[mask],
            r"$\epsilon$": circ.eps[mask],
            r"$\epsilon_r$": circ.eps_r[mask],
            # "Hue": labels,
            "Hue": labels2,
        }
    )
    expected_grid = sns.pairplot(df, hue="Hue", kind="hist", diag_kind="kde")

    assert_same_image(
        test_GalaxyPlotter_circularity_component_labels,
        format,
        test_grid,
        expected_grid,
    )
