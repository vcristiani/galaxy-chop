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

import pytest

import seaborn as sns

# =============================================================================
# UTILITIES
# =============================================================================


def image_paths(func, format):
    idir = _image_directories(test_plot_pairplot)[-1]
    idir.mkdir(parents=True, exist_ok=True)

    test = idir / f"{func.__name__}[{format}]-.{format}"
    expected = idir / f"{func.__name__}[{format}]-expected.{format}"

    return test, expected


def assert_same_image(test_func, format, test_img, ref_img, **kwargs):
    # Como la porqueria de pairplot no recibe ni ejes ni figuras no puedo
    # Usar las funciones de check_figures equals aca, asi que hay que hacer
    # todo a mano...

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


def test_plot_call_invalid_plot_kind(galaxy):
    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    with pytest.raises(ValueError):
        plotter("__call__")

    # not callable
    super(plot.GalaxyPlotter, plotter).__setattr__("zaraza", None)
    with pytest.raises(ValueError):
        plotter("zaraza")


@pytest.mark.parametrize("plot_kind", ["pairplot"])
def test_plot_call(galaxy, plot_kind):

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
@pytest.mark.parametrize("format", ["png", "pdf", "svg"])
def test_plot_pairplot(galaxy, format):

    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)
    test_grid = plotter.pairplot(attributes=["x", "y"])

    # EXPECTED
    df = gal.to_dataframe(attributes=["x", "y", "ptype"])
    expected_grid = sns.pairplot(
        data=df, hue="ptype", kind="hist", diag_kind="kde"
    )

    assert_same_image(test_plot_pairplot, format, test_grid, expected_grid)


@pytest.mark.slow
@pytest.mark.parametrize("format", ["png", "pdf", "svg"])
def test_plot_pairplot_external_labels(galaxy, format):

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
        test_plot_pairplot_external_labels, format, test_grid, expected_grid
    )


@pytest.mark.slow
@pytest.mark.parametrize("format", ["png", "pdf", "svg"])
def test_plot_dis(galaxy, format):
    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)
    test_grid = plotter.dis("x", "y", labels="ptype", ptypes=["gas"])

    # EXPECTED
    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    expected_grid = sns.displot(x="x", y="y", data=df, hue="ptype")

    assert_same_image(test_plot_dis, format, test_grid, expected_grid)


# =============================================================================
#
# =============================================================================


@pytest.mark.slow
@check_figures_equal()
def test_plot_scatter(galaxy, fig_test, fig_ref):

    gal = galaxy(seed=42)

    test_ax = fig_test.subplots()
    gal.plot.scatter("x", "y", labels="ptype", ptypes=["gas"], ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.scatterplot(data=df, x="x", y="y", hue="ptype", ax=exp_ax)


@pytest.mark.slow
@check_figures_equal()
def test_plot_hist(galaxy, fig_test, fig_ref):

    gal = galaxy(seed=42)

    test_ax = fig_test.subplots()
    gal.plot.hist("x", "y", labels="ptype", ptypes=["gas"], ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    df = gal.to_dataframe(ptypes=["gas"], attributes=["x", "y", "ptype"])
    sns.histplot(data=df, x="x", y="y", hue="ptype", ax=exp_ax)


@pytest.mark.slow
@check_figures_equal()
def test_plot_kde(galaxy, fig_test, fig_ref):

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
@check_figures_equal()
def test_plot_circ_hist(read_hdf5_galaxy, fig_test, fig_ref):

    gal = read_hdf5_galaxy("gal394242.h5")

    test_ax = fig_test.subplots()
    gal.plot.circ_hist(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    circ = utils.jcirc(gal)
    sns.histplot(circ.eps, ax=exp_ax)
    exp_ax.set_xlabel(r"$\epsilon$")


@pytest.mark.slow
@check_figures_equal()
def test_plot_circ_kde(read_hdf5_galaxy, fig_test, fig_ref):

    gal = read_hdf5_galaxy("gal394242.h5")

    test_ax = fig_test.subplots()
    gal.plot.circ_kde(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    circ = utils.jcirc(gal)
    sns.kdeplot(circ.eps, ax=exp_ax)
    exp_ax.set_xlabel(r"$\epsilon$")
