# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""test for galaxychop.constants"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import constants

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_get_component_style_named():
    assert (
        constants.get_component_style("disk") is constants.plot_config.disk
    )


def test_get_component_style_generic():
    style = constants.get_component_style("component_1")

    assert style.label == "Component 1"
    assert style.color == constants.GENERIC_COMPONENT_COLORS[1]
    # the same number always gets the same style
    assert constants.get_component_style("component_1") is style

    # the colors cycle when there are more components than colors
    n_colors = len(constants.GENERIC_COMPONENT_COLORS)
    wrapped = constants.get_component_style(f"component_{n_colors + 1}")
    assert wrapped.color == style.color


def test_get_component_style_generic_not_a_child():
    constants.get_component_style("component_0")
    assert all(
        child.label != "Component 0"
        for style in constants.plot_config.values()
        for child in style.childrens
    )


@pytest.mark.parametrize("key", ["unknown", "component_", "component_x"])
def test_get_component_style_invalid(key):
    with pytest.raises(KeyError):
        constants.get_component_style(key)


def test_make_component_styles_generic():
    names = {"stars": "stars", "component_0": "thin disk"}

    styles = constants.make_component_styles(names)

    assert styles["hue_order"] == ["stars", "thin disk"]
    assert styles["palette"] == {
        "stars": constants.plot_config.stars.color,
        "thin disk": constants.GENERIC_COMPONENT_COLORS[0],
    }
