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

from matplotlib.testing.decorators import check_figures_equal

import pytest

import seaborn as sns

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
@check_figures_equal()
def test_plot_pairplot(galaxy, fig_test, fig_ref):
    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=gal)

    # pairplot internamente llama a gcf (plt.subplots)
    # para crear la grilla de plots (Pairgrid) por lo que la unica
    # forma de pasarle la figura de testeo/referencia es con un mock
    axes = fig_test.subplots(2, 2)
    with mock.patch(
        "matplotlib.pyplot.subplots", return_value=(fig_test, axes)
    ):
        plotter.pairplot(attributes=["x", "y"])

    # EXPECTED
    axes = fig_ref.subplots(2, 2)

    df = gal.to_dataframe(columns=["x", "y", "ptype"])

    with mock.patch(
        "matplotlib.pyplot.subplots", return_value=(fig_ref, axes)
    ):
        sns.pairplot(data=df, hue="ptype", kind="hist", diag_kind="kde")
