# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test input data."""

# =============================================================================
# IMPORTS
# =============================================================================
from unittest import mock

from matplotlib import pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

import pytest

import seaborn as sns


from galaxychop import plot


# =============================================================================
# IO TESTS
# =============================================================================


@pytest.mark.parametrize("plot_kind", ["pairplot"])
def test_plot_call_heatmap(galaxy, plot_kind):

    gal = galaxy(seed=42)

    plotter = plot.GalaxyPlotter(galaxy=galaxy)

    method_name = f"galaxychop.plot.GalaxyPlotter.{plot_kind}"

    with mock.patch(method_name) as plot_method:
        plotter(plot_kind=plot_kind)

    plot_method.assert_called_once()
