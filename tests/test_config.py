# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Test for galaxychop.config"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import config

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_ComponentConf_immutability():
    """Test that _ComponentConf instances are immutable."""
    with pytest.raises(AttributeError):
        config.dark_matter.label = "New label"


def test_ComponentConf_childrens():
    """Test the `childrens` property of _ComponentConf."""
    assert isinstance(config.stars.childrens, frozenset)
    assert isinstance(config.disk.childrens, frozenset)
    assert isinstance(config.cold_disk.childrens, frozenset)
    assert isinstance(config.dark_matter.childrens, frozenset)

    assert config.cold_disk.childrens == frozenset()
    assert config.dark_matter.childrens == frozenset()


def test_config_hierarchy():
    """Test the parent references in the configuration (actual API)."""

    assert config.galaxy.parent is None

    assert config.disk.parent is config.galaxy
    assert config.stars.parent is config.galaxy
    assert config.dark_matter.parent is config.galaxy
    assert config.gas.parent is config.galaxy
    assert config.no_component.parent is config.galaxy

    assert config.spheroid.parent is config.stars
    assert config.cold_disk.parent is config.disk
