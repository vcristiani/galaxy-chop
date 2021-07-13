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

import pandas as pd

import pytest

# =============================================================================
# PARTICLE_SET TESTS
# =============================================================================


def test_ParticleSet_creation_with_potential(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=True
    )
    pset = core.ParticleSet(
        "foo",
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        softening=soft,
        potential=pot,
    )

    assert pset.ptype == "foo"
    assert np.all(pset.m.to_value() == m) and pset.m.unit == u.Msun
    assert np.all(pset.x.to_value() == x) and pset.x.unit == u.kpc
    assert np.all(pset.y.to_value() == y) and pset.y.unit == u.kpc
    assert np.all(pset.z.to_value() == z) and pset.z.unit == u.kpc
    assert np.all(pset.vx.to_value() == vx) and pset.vx.unit == (u.km / u.s)
    assert np.all(pset.vy.to_value() == vy) and pset.vy.unit == (u.km / u.s)
    assert np.all(pset.vz.to_value() == vz) and pset.vz.unit == (u.km / u.s)
    assert np.all(pset.softening == soft)

    assert pset.has_potential_
    assert np.all(pset.potential.to_value() == pot)
    assert pset.potential.unit == ((u.km / u.s) ** 2)


def test_ParticleSet_creation_without_potential(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=False
    )
    pset = core.ParticleSet(
        "foo",
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        softening=soft,
        potential=pot,
    )

    assert pset.ptype == "foo"
    assert np.all(pset.m.to_value() == m) and pset.m.unit == u.Msun
    assert np.all(pset.x.to_value() == x) and pset.x.unit == u.kpc
    assert np.all(pset.y.to_value() == y) and pset.y.unit == u.kpc
    assert np.all(pset.z.to_value() == z) and pset.z.unit == u.kpc
    assert np.all(pset.vx.to_value() == vx) and pset.vx.unit == (u.km / u.s)
    assert np.all(pset.vy.to_value() == vy) and pset.vy.unit == (u.km / u.s)
    assert np.all(pset.vz.to_value() == vz) and pset.vz.unit == (u.km / u.s)
    assert np.all(pset.softening == soft)

    assert not pset.has_potential_
    assert pset.potential is None


@pytest.mark.parametrize(
    "remove_one",
    ["m", "x", "y", "z", "vx", "vy", "vz", "potential"],
)
def test_ParticleSet_creation_bad_len(data_particleset, remove_one):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=True
    )
    params = dict(
        m=m, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, softening=soft, potential=pot
    )

    params[remove_one] = params[remove_one][1:]

    with pytest.raises(ValueError):
        core.ParticleSet("foo", **params)


def test_ParticleSet_len(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=True
    )

    pset = core.ParticleSet(
        "foo",
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        softening=soft,
        potential=pot,
    )

    assert (
        len(pset)
        == len(m)
        == len(x)
        == len(y)
        == len(z)
        == len(vx)
        == len(vy)
        == len(vz)
        == len(pot)
    )


@pytest.mark.parametrize("has_potential", [True, False])
def test_ParticleSet_to_dataframe(data_particleset, has_potential):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=has_potential
    )

    pset = core.ParticleSet(
        "foo",
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        softening=soft,
        potential=pot,
    )
    expected = pd.DataFrame(
        {
            "ptype": "foo",
            "m": m,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "softening": soft,
            "potential": pot if has_potential else np.full(len(pset), np.nan),
        }
    )
    df = pset.to_dataframe()

    assert df.equals(expected)


@pytest.mark.parametrize("has_potential", [True, False])
def test_ParticleSet_to_numpy(data_particleset, has_potential):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=has_potential
    )

    pset = core.ParticleSet(
        "foo",
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        softening=soft,
        potential=pot,
    )
    expected = np.column_stack(
        [
            m,
            x,
            y,
            z,
            vx,
            vy,
            vz,
            np.full(len(pset), soft),
            pot if has_potential else np.full(len(pset), np.nan),
        ]
    )
    arr = pset.to_numpy()

    assert np.array_equal(arr, expected, equal_nan=True)


@pytest.mark.parametrize("has_potential", [True, False])
def test_ParticleSet_repr(data_particleset, has_potential):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=has_potential
    )

    pset = core.ParticleSet(
        "foo",
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        softening=soft,
        potential=pot,
    )

    expected = f"ParticleSet(foo, size={len(m)}, softening={soft}, potentials={has_potential})"

    assert repr(pset) == expected


# =============================================================================
# TEST MK_GALAXY
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
        soft_s,
        pot_s,
        m_dm,
        x_dm,
        y_dm,
        z_dm,
        vx_dm,
        vy_dm,
        vz_dm,
        soft_dm,
        pot_dm,
        m_g,
        x_g,
        y_g,
        z_g,
        vx_g,
        vy_g,
        vz_g,
        soft_g,
        pot_g,
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
        softening_s=soft_s,
        softening_g=soft_g,
        softening_dm=soft_dm,
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
    np.testing.assert_array_equal(gal.stars.softening, soft_s)
    np.testing.assert_array_equal(gal.stars.potential.value, pot_s)

    np.testing.assert_array_equal(gal.dark_matter.m.value, m_dm)
    np.testing.assert_array_equal(gal.dark_matter.x.value, x_dm)
    np.testing.assert_array_equal(gal.dark_matter.y.value, y_dm)
    np.testing.assert_array_equal(gal.dark_matter.z.value, z_dm)
    np.testing.assert_array_equal(gal.dark_matter.vx.value, vx_dm)
    np.testing.assert_array_equal(gal.dark_matter.vy.value, vy_dm)
    np.testing.assert_array_equal(gal.dark_matter.vz.value, vz_dm)
    np.testing.assert_array_equal(gal.dark_matter.softening, soft_dm)
    np.testing.assert_array_equal(gal.dark_matter.potential.value, pot_dm)

    np.testing.assert_array_equal(gal.gas.m.value, m_g)
    np.testing.assert_array_equal(gal.gas.x.value, x_g)
    np.testing.assert_array_equal(gal.gas.y.value, y_g)
    np.testing.assert_array_equal(gal.gas.z.value, z_g)
    np.testing.assert_array_equal(gal.gas.vx.value, vx_g)
    np.testing.assert_array_equal(gal.gas.vy.value, vy_g)
    np.testing.assert_array_equal(gal.gas.vz.value, vz_g)
    np.testing.assert_array_equal(gal.gas.softening, soft_g)
    np.testing.assert_array_equal(gal.gas.potential.value, pot_g)
