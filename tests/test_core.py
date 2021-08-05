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
    assert np.all(pset.arr_.m == m) and pset.m.unit == u.Msun
    assert np.all(pset.arr_.x == x) and pset.x.unit == u.kpc
    assert np.all(pset.arr_.y == y) and pset.y.unit == u.kpc
    assert np.all(pset.arr_.z == z) and pset.z.unit == u.kpc
    assert np.all(pset.arr_.vx == vx) and pset.vx.unit == (u.km / u.s)
    assert np.all(pset.arr_.vy == vy) and pset.vy.unit == (u.km / u.s)
    assert np.all(pset.arr_.vz == vz) and pset.vz.unit == (u.km / u.s)
    assert np.all(pset.softening == soft)

    kinetic_energy = 0.5 * (vx ** 2 + vy ** 2 + vz ** 2)
    assert (
        np.all(pset.arr_.kinetic_energy_ == kinetic_energy)
        and pset.kinetic_energy_.unit == (u.km / u.s) ** 2
    )

    assert pset.has_potential_
    assert np.all(pset.arr_.potential == pot)
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
    assert np.all(pset.arr_.m == m) and pset.m.unit == u.Msun
    assert np.all(pset.arr_.x == x) and pset.x.unit == u.kpc
    assert np.all(pset.arr_.y == y) and pset.y.unit == u.kpc
    assert np.all(pset.arr_.z == z) and pset.z.unit == u.kpc
    assert np.all(pset.arr_.vx == vx) and pset.vx.unit == (u.km / u.s)
    assert np.all(pset.arr_.vy == vy) and pset.vy.unit == (u.km / u.s)
    assert np.all(pset.arr_.vz == vz) and pset.vz.unit == (u.km / u.s)
    assert np.all(pset.softening == soft)

    kinetic_energy = 0.5 * (vx ** 2 + vy ** 2 + vz ** 2)
    assert (
        np.all(pset.arr_.kinetic_energy_ == kinetic_energy)
        and pset.kinetic_energy_.unit == (u.km / u.s) ** 2
    )

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
            "kinetic_energy": 0.5 * (vx ** 2 + vy ** 2 + vz ** 2),
            "total_energy": 0.5 * (vx ** 2 + vy ** 2 + vz ** 2) + pot
            if has_potential
            else np.full(len(pset), np.nan),
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
            0.5 * (vx ** 2 + vy ** 2 + vz ** 2),
            0.5 * (vx ** 2 + vy ** 2 + vz ** 2) + pot
            if has_potential
            else np.full(len(pset), np.nan),
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

    expected = (
        f"ParticleSet(foo, size={len(m)}, "
        f"softening={soft}, potentials={has_potential})"
    )

    assert repr(pset) == expected


# =============================================================================
# TEST MK_GALAXY
# =============================================================================
@pytest.mark.parametrize("has_potential", [True, False])
def test_mkgakaxy(data_galaxy, has_potential):

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
    ) = data_galaxy(
        seed=42,
        stars_potential=has_potential,
        dm_potential=has_potential,
        gas_potential=has_potential,
    )

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
    assert np.all(gal.stars.arr_.m == m_s)
    assert np.all(gal.stars.arr_.x == x_s)
    assert np.all(gal.stars.arr_.y == y_s)
    assert np.all(gal.stars.arr_.z == z_s)
    assert np.all(gal.stars.arr_.vx == vx_s)
    assert np.all(gal.stars.arr_.vy == vy_s)
    assert np.all(gal.stars.arr_.vz == vz_s)
    assert np.all(gal.stars.softening == soft_s)

    assert np.all(gal.dark_matter.arr_.m == m_dm)
    assert np.all(gal.dark_matter.arr_.x == x_dm)
    assert np.all(gal.dark_matter.arr_.y == y_dm)
    assert np.all(gal.dark_matter.arr_.z == z_dm)
    assert np.all(gal.dark_matter.arr_.vx == vx_dm)
    assert np.all(gal.dark_matter.arr_.vy == vy_dm)
    assert np.all(gal.dark_matter.arr_.vz == vz_dm)
    assert np.all(gal.dark_matter.softening == soft_dm)

    assert np.all(gal.gas.arr_.m == m_g)
    assert np.all(gal.gas.arr_.x == x_g)
    assert np.all(gal.gas.arr_.y == y_g)
    assert np.all(gal.gas.arr_.z == z_g)
    assert np.all(gal.gas.arr_.vx == vx_g)
    assert np.all(gal.gas.arr_.vy == vy_g)
    assert np.all(gal.gas.arr_.vz == vz_g)
    assert np.all(gal.gas.softening == soft_g)
    assert gal.has_potential_ == has_potential

    if has_potential:
        assert (
            gal.stars.has_potential_
            and gal.dark_matter.has_potential_
            and gal.gas.has_potential_
        )
        assert np.all(gal.dark_matter.potential.value == pot_dm)
        assert np.all(gal.stars.potential.value == pot_s)
        assert np.all(gal.gas.potential.value == pot_g)

    else:
        assert not (
            gal.stars.has_potential_
            or gal.dark_matter.has_potential_
            or gal.gas.has_potential_
        )
        assert gal.dark_matter.potential is None
        assert gal.stars.potential is None
        assert gal.gas.potential is None


@pytest.mark.parametrize("remove_potential", ["pot_s", "pot_g", "pot_dm"])
def test_mkgakaxy_missing_potential(data_galaxy, remove_potential):

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

    params = {
        "m_s": m_s,
        "x_s": x_s,
        "y_s": y_s,
        "z_s": z_s,
        "vx_s": vx_s,
        "vy_s": vy_s,
        "vz_s": vz_s,
        "m_dm": m_dm,
        "x_dm": x_dm,
        "y_dm": y_dm,
        "z_dm": z_dm,
        "vx_dm": vx_dm,
        "vy_dm": vy_dm,
        "vz_dm": vz_dm,
        "m_g": m_g,
        "x_g": x_g,
        "y_g": y_g,
        "z_g": z_g,
        "vx_g": vx_g,
        "vy_g": vy_g,
        "vz_g": vz_g,
        "softening_s": soft_s,
        "softening_g": soft_g,
        "softening_dm": soft_dm,
        "pot_s": pot_s,
        "pot_g": pot_g,
        "pot_dm": pot_dm,
    }

    params[remove_potential] = None

    with pytest.raises(ValueError):
        core.mkgalaxy(**params)


# =============================================================================
# KINECTIC ENERGY
# =============================================================================


def test_Galaxy_kinectic_energy(galaxy):
    gal = galaxy(seed=42)
    gke = gal.kinetic_energy_
    assert np.all(gke[0] == gal.stars.kinetic_energy_)
    assert np.all(gke[1] == gal.dark_matter.kinetic_energy_)
    assert np.all(gke[2] == gal.gas.kinetic_energy_)


# =============================================================================
#   POTENTIAL ENERGY
# =============================================================================


def test_Galaxy_potential_energy_already_calculated(galaxy):
    gal = galaxy(seed=42)
    with pytest.raises(ValueError):
        gal.potential_energy()


@pytest.mark.xfail
def test_Galaxy_potential_energy(galaxy):
    gal = galaxy(
        seed=42, stars_potential=False, dm_potential=False, gas_potential=False
    )
    pgal = gal.potential_energy()
    assert pgal
