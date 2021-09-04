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
from galaxychop import utils

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
        potential_s,
        m_dm,
        x_dm,
        y_dm,
        z_dm,
        vx_dm,
        vy_dm,
        vz_dm,
        soft_dm,
        potential_dm,
        m_g,
        x_g,
        y_g,
        z_g,
        vx_g,
        vy_g,
        vz_g,
        soft_g,
        potential_g,
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
        potential_s=potential_s,
        potential_g=potential_g,
        potential_dm=potential_dm,
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
        assert np.all(gal.dark_matter.potential.value == potential_dm)
        assert np.all(gal.stars.potential.value == potential_s)
        assert np.all(gal.gas.potential.value == potential_g)

    else:
        assert not (
            gal.stars.has_potential_
            or gal.dark_matter.has_potential_
            or gal.gas.has_potential_
        )
        assert gal.dark_matter.potential is None
        assert gal.stars.potential is None
        assert gal.gas.potential is None


@pytest.mark.parametrize(
    "remove_potential", ["potential_s", "potential_g", "potential_dm"]
)
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
        potential_s,
        m_dm,
        x_dm,
        y_dm,
        z_dm,
        vx_dm,
        vy_dm,
        vz_dm,
        soft_dm,
        potential_dm,
        m_g,
        x_g,
        y_g,
        z_g,
        vx_g,
        vy_g,
        vz_g,
        soft_g,
        potential_g,
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
        "potential_s": potential_s,
        "potential_g": potential_g,
        "potential_dm": potential_dm,
    }

    params[remove_potential] = None

    with pytest.raises(ValueError):
        core.mkgalaxy(**params)


# =============================================================================
# AS KWARGS
# =============================================================================


def test_galaxy_as_kwargs(data_galaxy):
    (
        m_s,
        x_s,
        y_s,
        z_s,
        vx_s,
        vy_s,
        vz_s,
        soft_s,
        potential_s,
        m_dm,
        x_dm,
        y_dm,
        z_dm,
        vx_dm,
        vy_dm,
        vz_dm,
        soft_dm,
        potential_dm,
        m_g,
        x_g,
        y_g,
        z_g,
        vx_g,
        vy_g,
        vz_g,
        soft_g,
        potential_g,
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
        potential_s=potential_s,
        potential_g=potential_g,
        potential_dm=potential_dm,
    )

    gkwargs = core.galaxy_as_kwargs(gal)

    assert np.all(gkwargs["m_s"].to_value() == m_s)
    assert np.all(gkwargs["x_s"].to_value() == x_s)
    assert np.all(gkwargs["y_s"].to_value() == y_s)
    assert np.all(gkwargs["z_s"].to_value() == z_s)
    assert np.all(gkwargs["vx_s"].to_value() == vx_s)
    assert np.all(gkwargs["vy_s"].to_value() == vy_s)
    assert np.all(gkwargs["vz_s"].to_value() == vz_s)
    assert np.all(gkwargs["m_dm"].to_value() == m_dm)
    assert np.all(gkwargs["x_dm"].to_value() == x_dm)
    assert np.all(gkwargs["y_dm"].to_value() == y_dm)
    assert np.all(gkwargs["z_dm"].to_value() == z_dm)
    assert np.all(gkwargs["vx_dm"].to_value() == vx_dm)
    assert np.all(gkwargs["vy_dm"].to_value() == vy_dm)
    assert np.all(gkwargs["vz_dm"].to_value() == vz_dm)
    assert np.all(gkwargs["m_g"].to_value() == m_g)
    assert np.all(gkwargs["x_g"].to_value() == x_g)
    assert np.all(gkwargs["y_g"].to_value() == y_g)
    assert np.all(gkwargs["z_g"].to_value() == z_g)
    assert np.all(gkwargs["vx_g"].to_value() == vx_g)
    assert np.all(gkwargs["vy_g"].to_value() == vy_g)
    assert np.all(gkwargs["vz_g"].to_value() == vz_g)
    assert np.all(gkwargs["softening_s"] == soft_s)
    assert np.all(gkwargs["softening_g"] == soft_g)
    assert np.all(gkwargs["softening_dm"] == soft_dm)
    assert np.all(gkwargs["potential_s"].to_value() == potential_s)
    assert np.all(gkwargs["potential_g"].to_value() == potential_g)
    assert np.all(gkwargs["potential_dm"].to_value() == potential_dm)


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
#   TOTAL ENERGY
# =============================================================================


def test_Galaxy_total_energy(galaxy):
    gal = galaxy(seed=42)
    gte = gal.total_energy_
    assert np.all(gte[0] == gal.stars.total_energy_)
    assert np.all(gte[1] == gal.dark_matter.total_energy_)
    assert np.all(gte[2] == gal.gas.total_energy_)


def test_Galaxy_energy(galaxy):
    gal = galaxy(seed=42)
    energy = gal.total_energy_
    gke = gal.kinetic_energy_
    gpe = gal.potential_energy_

    assert np.all(gke[0] + gpe[0] == energy[0])
    assert np.all(gke[1] + gpe[1] == energy[1])
    assert np.all(gke[2] + gpe[2] == energy[2])


# =============================================================================
#   ANGULAR MOMENTUM
# =============================================================================


@pytest.mark.xfail
def test_center_existence(galaxy):
    gal = galaxy(seed=42)

    gx_c = utils.center(gal.x, gal.y, gal.z, gal.potential)
    #    gal.stars.arr_.x,
    #    gal.stars.arr_.y,
    #    gal.stars.arr_.z,
    #    gal.dark_matter.arr_.m,
    #    gal.dark_matter.arr_.x,
    #    gal.dark_matter.arr_.y,
    #    gal.dark_matter.arr_.z,
    #    gal.gas.arr_.m,
    #    gal.gas.arr_.x,
    #    gal.gas.arr_.y,
    #    gal.gas.arr_.z,
    #    )

    x_gal = np.hstack((gx_c[0], gx_c[3], gx_c[6]))
    y_gal = np.hstack((gx_c[1], gx_c[4], gx_c[7]))
    z_gal = np.hstack((gx_c[2], gx_c[5], gx_c[8]))

    pos_gal = np.vstack((x_gal, y_gal, z_gal))

    assert len(np.where(~pos_gal.any(axis=0))) == 1


@pytest.mark.xfail
def test_angular_momentum_outputs(galaxy):
    """Test object."""
    gal = galaxy(seed=42)
    gam = gal.angular_momentum()

    longitude = len(gam.stars.x) + len(gam.dark_matter.x) + len(gam.gas.x)
    assert np.shape(gam.J_part.value) == (3, longitude)
