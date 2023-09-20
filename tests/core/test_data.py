# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test input core."""

# =============================================================================
# IMPORTS
# =============================================================================

from io import BytesIO

import astropy.units as u

from galaxychop import core, io

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# PARTICLESET TYPE TESTS
# =============================================================================


def test_ParticleSetType():
    assert (
        core.ParticleSetType.mktype("stars")
        == core.ParticleSetType.mktype("STARS")
        == core.ParticleSetType.mktype(0)
        == core.ParticleSetType.mktype(core.ParticleSetType.STARS)
        == core.ParticleSetType.STARS
    ) and core.ParticleSetType.STARS.humanize() == "stars"

    assert (
        core.ParticleSetType.mktype("dark_matter")
        == core.ParticleSetType.mktype("DARK_MATTER")
        == core.ParticleSetType.mktype(1)
        == core.ParticleSetType.mktype(core.ParticleSetType.DARK_MATTER)
        == core.ParticleSetType.DARK_MATTER
    ) and core.ParticleSetType.DARK_MATTER.humanize() == "dark_matter"

    assert (
        core.ParticleSetType.mktype("gas")
        == core.ParticleSetType.mktype("GAS")
        == core.ParticleSetType.mktype(2)
        == core.ParticleSetType.mktype(core.ParticleSetType.GAS)
        == core.ParticleSetType.GAS
    ) and core.ParticleSetType.GAS.humanize() == "gas"

    with pytest.raises(ValueError):
        core.ParticleSetType.mktype(43)
    with pytest.raises(ValueError):
        core.ParticleSetType.mktype("foo")
    with pytest.raises(ValueError):
        core.ParticleSetType.mktype(None)


# =============================================================================
# PARTICLE_SET TESTS
# =============================================================================


def test_ParticleSet_creation_with_potential(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=True
    )
    pset = core.ParticleSet(
        core.ParticleSetType.STARS,
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

    assert pset.ptype == core.ParticleSetType.STARS
    assert np.all(pset.arr_.m == m) and pset.m.unit == u.Msun
    assert np.all(pset.arr_.x == x) and pset.x.unit == u.kpc
    assert np.all(pset.arr_.y == y) and pset.y.unit == u.kpc
    assert np.all(pset.arr_.z == z) and pset.z.unit == u.kpc
    assert np.all(pset.arr_.vx == vx) and pset.vx.unit == (u.km / u.s)
    assert np.all(pset.arr_.vy == vy) and pset.vy.unit == (u.km / u.s)
    assert np.all(pset.arr_.vz == vz) and pset.vz.unit == (u.km / u.s)
    assert np.all(pset.softening == soft)

    kinetic_energy = 0.5 * (vx**2 + vy**2 + vz**2)
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
        core.ParticleSetType.STARS,
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

    assert pset.ptype == core.ParticleSetType.STARS
    assert np.all(pset.arr_.m == m) and pset.m.unit == u.Msun
    assert np.all(pset.arr_.x == x) and pset.x.unit == u.kpc
    assert np.all(pset.arr_.y == y) and pset.y.unit == u.kpc
    assert np.all(pset.arr_.z == z) and pset.z.unit == u.kpc
    assert np.all(pset.arr_.vx == vx) and pset.vx.unit == (u.km / u.s)
    assert np.all(pset.arr_.vy == vy) and pset.vy.unit == (u.km / u.s)
    assert np.all(pset.arr_.vz == vz) and pset.vz.unit == (u.km / u.s)
    assert np.all(pset.softening == soft)

    kinetic_energy = 0.5 * (vx**2 + vy**2 + vz**2)
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
        core.ParticleSet(core.ParticleSetType.STARS, **params)


def test_ParticleSet_len(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=True
    )

    pset = core.ParticleSet(
        core.ParticleSetType.STARS,
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
        core.ParticleSetType.STARS,
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
            "ptype": core.ParticleSetType.STARS.humanize(),
            "ptypev": core.ParticleSetType.STARS.value,
            "m": m,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "softening": soft,
            "potential": pot if has_potential else np.full(len(pset), np.nan),
            "kinetic_energy": 0.5 * (vx**2 + vy**2 + vz**2),
            "total_energy": (
                0.5 * (vx**2 + vy**2 + vz**2) + pot
                if has_potential
                else np.full(len(pset), np.nan)
            ),
            "Jx": y * vz - z * vy,
            "Jy": z * vx - x * vz,
            "Jz": x * vy - y * vx,
        }
    )
    df = pset.to_dataframe()

    assert df.equals(expected)


def test_ParticleSet_to_dataframe_no_potential(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=False
    )

    pset = core.ParticleSet(
        core.ParticleSetType.STARS,
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
            "potential": np.full(len(pset), np.nan),
            "total_energy": np.full(len(pset), np.nan),
        }
    )

    df = pset.to_dataframe(attributes=["potential", "total_energy"])

    assert df.equals(expected)


@pytest.mark.parametrize("has_potential", [True, False])
def test_ParticleSet_repr(data_particleset, has_potential):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=has_potential
    )

    pset = core.ParticleSet(
        core.ParticleSetType.STARS,
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
        f"<ParticleSet 'STARS', size={len(m)}, "
        f"softening={soft}, potentials={has_potential}>"
    )

    assert repr(pset) == expected


def test_ParticleSet_angular_momentum(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(seed=42)

    pset = core.ParticleSet(
        core.ParticleSetType.STARS,
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

    assert np.all(pset.Jx_ == pset.angular_momentum_[0])
    assert np.all(pset.Jy_ == pset.angular_momentum_[1])
    assert np.all(pset.Jz_ == pset.angular_momentum_[2])
    assert np.all(
        pset.Jx_.unit
        == pset.Jy_.unit
        == pset.Jz_.unit
        == pset.angular_momentum_.unit
        == (u.kpc * u.km / u.s)
    )


def test_ParticleSet_to_dict(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(seed=42)

    pset = core.ParticleSet(
        core.ParticleSetType.STARS,
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

    pset_dict = pset.to_dict()

    np.testing.assert_equal(
        pset_dict["ptype"],
        [core.ParticleSetType.STARS.humanize()] * len(pset),
    )
    np.testing.assert_equal(
        pset_dict["ptypev"],
        [core.ParticleSetType.STARS.value] * len(pset),
    )

    np.testing.assert_allclose(pset_dict["m"], m)
    np.testing.assert_allclose(pset_dict["m"], pset.m.to_value())
    np.testing.assert_allclose(pset_dict["x"], x)
    np.testing.assert_allclose(pset_dict["x"], pset.x.to_value())
    np.testing.assert_allclose(pset_dict["y"], y)
    np.testing.assert_allclose(pset_dict["y"], pset.y.to_value())
    np.testing.assert_allclose(pset_dict["z"], z)
    np.testing.assert_allclose(pset_dict["z"], pset.z.to_value())
    np.testing.assert_allclose(pset_dict["vx"], vx)
    np.testing.assert_allclose(pset_dict["vx"], pset.vx.to_value())
    np.testing.assert_allclose(pset_dict["vy"], vy)
    np.testing.assert_allclose(pset_dict["vy"], pset.vy.to_value())
    np.testing.assert_allclose(pset_dict["vz"], vz)
    np.testing.assert_allclose(pset_dict["vz"], pset.vz.to_value())
    np.testing.assert_allclose(pset_dict["softening"], soft)
    np.testing.assert_allclose(pset_dict["softening"], pset.softening)
    np.testing.assert_allclose(pset_dict["potential"], pot)
    np.testing.assert_allclose(
        pset_dict["potential"], pset.potential.to_value()
    )

    np.testing.assert_allclose(
        pset_dict["kinetic_energy"], pset.kinetic_energy_.to_value()
    )
    np.testing.assert_allclose(
        pset_dict["total_energy"], pset.total_energy_.to_value()
    )
    np.testing.assert_allclose(pset_dict["Jx"], pset.Jx_.to_value())
    np.testing.assert_allclose(pset_dict["Jy"], pset.Jy_.to_value())
    np.testing.assert_allclose(pset_dict["Jz"], pset.Jz_.to_value())


def test_ParticleSet_copy(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(seed=42)

    pset = core.ParticleSet(
        core.ParticleSetType.STARS,
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

    pset_copy = pset.copy()

    assert pset_copy is not pset

    assert pset_copy.ptype is pset_copy.ptype

    np.testing.assert_equal(pset_copy.m, pset.m)
    assert pset_copy.m is not pset.m
    np.testing.assert_equal(pset_copy.x, pset.x)
    assert pset_copy.x is not pset.x
    np.testing.assert_equal(pset_copy.y, pset.y)
    assert pset_copy.y is not pset.y
    np.testing.assert_equal(pset_copy.z, pset.z)
    assert pset_copy.z is not pset.z
    np.testing.assert_equal(pset_copy.vx, pset.vx)
    assert pset_copy.vx is not pset.vx
    np.testing.assert_equal(pset_copy.vy, pset.vy)
    assert pset_copy.vy is not pset.vy
    np.testing.assert_equal(pset_copy.vz, pset.vz)
    assert pset_copy.vz is not pset.vz
    np.testing.assert_equal(pset_copy.potential, pset.potential)
    assert pset_copy.potential is not pset.potential
    np.testing.assert_equal(pset_copy.softening, pset.softening)

    np.testing.assert_equal(pset_copy.kinetic_energy_, pset.kinetic_energy_)
    assert pset_copy.kinetic_energy_ is not pset.kinetic_energy_
    np.testing.assert_equal(pset_copy.total_energy_, pset.total_energy_)
    assert pset_copy.total_energy_ is not pset.total_energy_
    np.testing.assert_equal(pset_copy.Jx_, pset.Jx_)
    assert pset_copy.Jx_ is not pset.Jx_
    np.testing.assert_equal(pset_copy.Jy_, pset.Jy_)
    assert pset_copy.Jy_ is not pset.Jy_
    np.testing.assert_equal(pset_copy.Jz_, pset.Jz_)
    assert pset_copy.Jz_ is not pset.Jz_


# =============================================================================
# TEST GALAXY MANUAL
# =============================================================================


def test_Galaxy_invalid_ParticleSetType(data_particleset):
    bad_types = {
        "stars": core.ParticleSetType.DARK_MATTER,  # WRONG
        "dark_matter": core.ParticleSetType.GAS,
        "gas": core.ParticleSetType.STARS,
    }

    gal_kwargs = {}
    for pname, ptype in bad_types.items():
        m, x, y, z, vx, vy, vz, soft, pot = data_particleset()
        pset = core.ParticleSet(
            ptype,
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
        gal_kwargs[pname] = pset

    with pytest.raises(TypeError):
        core.Galaxy(**gal_kwargs)


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
    assert len(gal) == len(gal.stars) + len(gal.dark_matter) + len(gal.gas)

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
# PLOTTER
# =============================================================================


def test_Galaxy_repr(galaxy):
    gal = galaxy(
        stars_min=100,
        stars_max=100,
        dm_min=100,
        dm_max=100,
        gas_min=100,
        gas_max=100,
    )

    expected = "<Galaxy stars=100, dark_matter=100, gas=100, potential=True>"
    assert repr(gal) == expected


# =============================================================================
# PLOTTER
# =============================================================================


def test_Galaxy_plot(galaxy):
    gal = galaxy()
    assert isinstance(gal.plot, core.plot.GalaxyPlotter)
    assert gal.plot._galaxy is gal


# =============================================================================
# AS DISASSEMBLE
# =============================================================================


def test_galaxy_disassemble(data_galaxy):
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

    gkwargs = gal.disassemble()

    assert np.all(gkwargs["m_s"] == m_s)
    assert np.all(gkwargs["x_s"] == x_s)
    assert np.all(gkwargs["y_s"] == y_s)
    assert np.all(gkwargs["z_s"] == z_s)
    assert np.all(gkwargs["vx_s"] == vx_s)
    assert np.all(gkwargs["vy_s"] == vy_s)
    assert np.all(gkwargs["vz_s"] == vz_s)
    assert np.all(gkwargs["m_dm"] == m_dm)
    assert np.all(gkwargs["x_dm"] == x_dm)
    assert np.all(gkwargs["y_dm"] == y_dm)
    assert np.all(gkwargs["z_dm"] == z_dm)
    assert np.all(gkwargs["vx_dm"] == vx_dm)
    assert np.all(gkwargs["vy_dm"] == vy_dm)
    assert np.all(gkwargs["vz_dm"] == vz_dm)
    assert np.all(gkwargs["m_g"] == m_g)
    assert np.all(gkwargs["x_g"] == x_g)
    assert np.all(gkwargs["y_g"] == y_g)
    assert np.all(gkwargs["z_g"] == z_g)
    assert np.all(gkwargs["vx_g"] == vx_g)
    assert np.all(gkwargs["vy_g"] == vy_g)
    assert np.all(gkwargs["vz_g"] == vz_g)
    assert np.all(gkwargs["softening_s"] == soft_s)
    assert np.all(gkwargs["softening_g"] == soft_g)
    assert np.all(gkwargs["softening_dm"] == soft_dm)
    assert np.all(gkwargs["potential_s"] == potential_s)
    assert np.all(gkwargs["potential_g"] == potential_g)
    assert np.all(gkwargs["potential_dm"] == potential_dm)


# =============================================================================
# TO HDF5
# =============================================================================


def test_Galaxy_to_hdf5(galaxy):
    gal = galaxy()

    stream = BytesIO()
    gal.to_hdf5(stream)
    stream.seek(0)
    result = io.read_hdf5(stream)

    stored_attributes = ["m", "x", "y", "z", "vx", "vy", "vz"]

    result_df = result.to_dataframe(attributes=stored_attributes)
    expected_df = gal.to_dataframe(attributes=stored_attributes)

    pd.testing.assert_frame_equal(result_df, expected_df)


# =============================================================================
# TO DICT
# =============================================================================


def assert_pset_dict_equals(result, expected):
    assert result.keys() == expected.keys()
    for key in result:
        assert np.array_equal(result[key], expected[key])


def test_Galaxy_to_dict(galaxy):
    gal = galaxy()
    gal_dict = gal.to_dict()

    assert_pset_dict_equals(gal_dict["stars"], gal.stars.to_dict())
    assert_pset_dict_equals(gal_dict["dark_matter"], gal.dark_matter.to_dict())
    assert_pset_dict_equals(gal_dict["gas"], gal.gas.to_dict())


# =============================================================================
# TO copy
# =============================================================================


def assert_pset_equals(result, expected):
    assert isinstance(result, core.ParticleSet)

    result_dict = result.to_dict()
    expected_dict = expected.to_dict()

    assert_pset_dict_equals(result_dict, expected_dict)


def test_Galaxy_to_copy(galaxy):
    gal = galaxy()
    gal_copy = gal.copy()

    assert gal_copy is not gal

    assert gal_copy.stars is not gal.stars
    assert_pset_equals(gal_copy.stars, gal.stars)

    assert gal_copy.dark_matter is not gal.dark_matter
    assert_pset_equals(gal_copy.dark_matter, gal.dark_matter)

    assert gal_copy.gas is not gal.gas
    assert_pset_equals(gal_copy.gas, gal.gas)


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
# POTENTIAL ENERGY
# =============================================================================


def test_Galaxy_potential_energy(galaxy):
    gal = galaxy(seed=42)
    s_pot, dm_pot, gas_pot = gal.potential_energy_
    assert s_pot.unit == dm_pot.unit == gas_pot.unit == ((u.km / u.s) ** 2)
    assert np.all(gal.stars.potential.to_value() == s_pot.to_value())
    assert np.all(gal.dark_matter.potential.to_value() == dm_pot.to_value())
    assert np.all(gal.gas.potential.to_value() == gas_pot.to_value())


def test_Galaxy_potential_energy_without_potential(galaxy):
    gal = galaxy(
        stars_potential=False, dm_potential=False, gas_potential=False
    )
    assert gal.potential_energy_ is None


# =============================================================================
# TOTAL ENERGY
# =============================================================================


def test_Galaxy_total_energy(galaxy):
    gal = galaxy(seed=42)
    gte = gal.total_energy_
    assert np.all(gte[0] == gal.stars.total_energy_)
    assert np.all(gte[1] == gal.dark_matter.total_energy_)
    assert np.all(gte[2] == gal.gas.total_energy_)


def test_Galaxy_all_energies(galaxy):
    gal = galaxy(seed=42)
    energy = gal.total_energy_
    gke = gal.kinetic_energy_
    gpe = gal.potential_energy_

    assert np.all(gke[0] + gpe[0] == energy[0])
    assert np.all(gke[1] + gpe[1] == energy[1])
    assert np.all(gke[2] + gpe[2] == energy[2])


def test_Galaxy_total_energy_without_potential(galaxy):
    gal = galaxy(
        stars_potential=False, dm_potential=False, gas_potential=False
    )
    assert gal.total_energy_ is None


# =============================================================================
#   ANGULAR MOMENTUM
# =============================================================================


def test_Galaxy_angular_momentum(galaxy):
    gal = galaxy(seed=42)
    gam = gal.angular_momentum_
    assert np.all(gam[0] == gal.stars.angular_momentum_)
    assert np.all(gam[1] == gal.dark_matter.angular_momentum_)
    assert np.all(gam[2] == gal.gas.angular_momentum_)


# =============================================================================
# STELLAR
# =============================================================================


def test_Galaxy_stellar_dynamics(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    sd_from_gal = gal.stellar_dynamics()
    sd_from_func = core.sdynamics.stellar_dynamics(gal)

    np.testing.assert_array_equal(sd_from_gal.x, sd_from_func.x)
    np.testing.assert_array_equal(sd_from_gal.y, sd_from_func.y)

    sd_from_gal_dict = sd_from_gal.to_dict()
    sd_from_func_dict = sd_from_func.to_dict()

    assert sd_from_func_dict.keys() == sd_from_gal_dict.keys()
    for k, gv in sd_from_gal_dict.items():
        fv = sd_from_func_dict[k]
        np.testing.assert_array_equal(gv, fv)

    np.testing.assert_array_equal(gv, fv)
