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

import pytest

from galaxychop import config

# =============================================================================
# TESTS
# =============================================================================


def test_ComponentConf_immutability():
    """Test that _ComponentConf instances are immutable."""
    with pytest.raises(AttributeError):
        config.dm.label = "New label"


def test_ComponentConf_childrens():
    """Test the `childrens` property of _ComponentConf."""
    # Stars has two direct children: Disk and Spheroid
    assert config.stars.childrens == frozenset([config.disk, config.spheroid])

    # Disk has three direct children
    assert config.disk.childrens == frozenset(
        [config.cold_disk, config.warm_disk, config.bar]
    )

    # A leaf component like cold_disk has no children
    assert config.cold_disk.childrens == frozenset()

    # A root component like dm has no children either
    assert config.dm.childrens == frozenset()


def test_config_hierarchy():
    """Test the parent-child references in the configuration."""
    # Test a two-level hierarchy
    assert config.disk.parent is config.stars
    assert config.spheroid.parent is config.stars

    # Test a three-level hierarchy
    assert config.cold_disk.parent is config.disk
    assert config.cold_disk.parent.parent is config.stars

    # Test root components
    assert config.stars.parent is None
    assert config.dm.parent is None
