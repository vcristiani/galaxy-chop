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

from galaxychop import plot


from matplotlib.testing.decorators import _image_directories, compare_images

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


# =============================================================================
# IO TESTS
# =============================================================================


@pytest.mark.parametrize("plot_kind", ["pairplot"])
def test_plot_call_heatmap(galaxy, plot_kind):

    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    method_name = f"galaxychop.plot.GalaxyPlotter.{plot_kind}"

    with mock.patch(method_name) as plot_method:
        plotter(plot_kind=plot_kind)

    plot_method.assert_called_once()


@pytest.mark.slow
@pytest.mark.parametrize("format", ["png", "pdf", "svg"])
def test_plot_pairplot(galaxy, format):
    # Como la porqueria de pairplot no recibe ni ejes ni figuras no puedo
    # Usar las funciones de check_figures equals aca, asi que hay que hacer
    # todo a mano...

    test_path, ref_path = image_paths(test_plot_pairplot, format)

    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    g = plotter.pairplot(attributes=["x", "y"])
    g.savefig(test_path)

    # EXPECTED
    df = gal.to_dataframe(attributes=["x", "y", "ptype"])
    g = sns.pairplot(data=df, hue="ptype", kind="hist", diag_kind="kde")
    g.savefig(ref_path)

    result = compare_images(test_path, ref_path, 0)
    if result:
        pytest.fail(result)
