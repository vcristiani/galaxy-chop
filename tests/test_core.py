# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test input data."""

# =============================================================================
# IMPORTS
# =============================================================================

import astropy.units as u

from galaxychop import core

import numpy as np

# =============================================================================
# TESTS
# =============================================================================


def test_mkgakaxy(data_galaxy):

    (
        m_s,
        x_s,
        y_s,
        z_s,
        vx_s,
        vy_s,
        vz_s,
        m_dm,
        x_dm,
        y_dm,
        z_dm,
        vx_dm,
        vy_dm,
        vz_dm,
        m_g,
        x_g,
        y_g,
        z_g,
        vx_g,
        vy_g,
        vz_g,
        softening_s,
        softening_g,
        softening_dm,
        pot_s,
        pot_g,
        pot_dm,
    ) = data_galaxy(seed=42)

    gal = core.mkgalaxy(
        m_s=m_s,
        x_s=x_s,
        y_s=y_s,
        z_s=z_s,
        vx_s=vx_s,
        vy_s=vy_s,
        vz_s=vz_s,
        m_dm=m_dm,
        x_dm=x_dm,
        y_dm=y_dm,
        z_dm=z_dm,
        vx_dm=vx_dm,
        vy_dm=vy_dm,
        vz_dm=vz_dm,
        m_g=m_g,
        x_g=x_g,
        y_g=y_g,
        z_g=z_g,
        vx_g=vx_g,
        vy_g=vy_g,
        vz_g=vz_g,
        softening_s=softening_s,
        softening_g=softening_g,
        softening_dm=softening_dm,
        pot_s=pot_s,
        pot_g=pot_g,
        pot_dm=pot_dm,
    )
    np.testing.assert_array_equal(gal.stars.m.value, m_s)
    np.testing.assert_array_equal(gal.stars.x.value, x_s)
    np.testing.assert_array_equal(gal.stars.y.value, y_s)
    np.testing.assert_array_equal(gal.stars.z.value, z_s)
    np.testing.assert_array_equal(gal.stars.vx.value, vx_s)
    np.testing.assert_array_equal(gal.stars.vy.value, vy_s)
    np.testing.assert_array_equal(gal.stars.vz.value, vz_s)
    np.testing.assert_array_equal(gal.stars.softening, softening_s)
    np.testing.assert_array_equal(gal.stars.potential.value, pot_s)

    np.testing.assert_array_equal(gal.dark_matter.m.value, m_dm)
    np.testing.assert_array_equal(gal.dark_matter.x.value, x_dm)
    np.testing.assert_array_equal(gal.dark_matter.y.value, y_dm)
    np.testing.assert_array_equal(gal.dark_matter.z.value, z_dm)
    np.testing.assert_array_equal(gal.dark_matter.vx.value, vx_dm)
    np.testing.assert_array_equal(gal.dark_matter.vy.value, vy_dm)
    np.testing.assert_array_equal(gal.dark_matter.vz.value, vz_dm)
    np.testing.assert_array_equal(gal.dark_matter.softening, softening_dm)
    np.testing.assert_array_equal(gal.dark_matter.potential.value, pot_dm)

    np.testing.assert_array_equal(gal.gas.m.value, m_g)
    np.testing.assert_array_equal(gal.gas.x.value, x_g)
    np.testing.assert_array_equal(gal.gas.y.value, y_g)
    np.testing.assert_array_equal(gal.gas.z.value, z_g)
    np.testing.assert_array_equal(gal.gas.vx.value, vx_g)
    np.testing.assert_array_equal(gal.gas.vy.value, vy_g)
    np.testing.assert_array_equal(gal.gas.vz.value, vz_g)
    np.testing.assert_array_equal(gal.gas.softening, softening_g)
    np.testing.assert_array_equal(gal.gas.potential.value, pot_g)

    assert len(m_s) == len(x_s) == len(y_s) == len(z_s)
    assert len(m_s) == len(vx_s) == len(vy_s) == len(vz_s)
    assert len(m_s) == len(gal.stars.m)
    assert len(m_s) == len(gal.stars.x) == len(gal.stars.y) == len(gal.stars.z)
    assert len(m_s) == len(gal.stars.vx)
    assert len(m_s) == len(gal.stars.vy) == len(gal.stars.vz)

    assert isinstance(gal.stars.m, u.Quantity)
    assert isinstance(gal.stars.x, u.Quantity)
    assert isinstance(gal.stars.y, u.Quantity)
    assert isinstance(gal.stars.z, u.Quantity)
    assert isinstance(gal.stars.vx, u.Quantity)
    assert isinstance(gal.stars.vy, u.Quantity)
    assert isinstance(gal.stars.vz, u.Quantity)
    assert isinstance(gal.stars.potential, u.Quantity)
