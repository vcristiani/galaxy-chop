# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
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
        config.dm.label = "New label"


def test_ComponentConf_childrens():
    """Test the `childrens` property of _ComponentConf."""
    assert isinstance(config.stars.childrens, frozenset)
    assert isinstance(config.disk.childrens, frozenset)
    assert isinstance(config.cold_disk.childrens, frozenset)
    assert isinstance(config.dm.childrens, frozenset)

    assert config.cold_disk.childrens == frozenset()
    assert config.dm.childrens == frozenset()


def test_config_hierarchy():
    """Test the parent references in the configuration (actual API)."""

    assert config.disk.parent is config.stars or config.disk.parent is None
    assert config.spheroid.parent is config.stars

    assert config.cold_disk.parent is config.disk

    assert config.stars.parent is None
    assert config.dm.parent is None
