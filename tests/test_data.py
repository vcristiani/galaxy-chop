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

from galaxychop import data, plot

import numpy as np

import pandas as pd

import pytest

# =============================================================================
# PARTICLESET TYPE TESTS
# =============================================================================


def test_ParticleSetType():
    assert (
        data.ParticleSetType.mktype("stars")
        == data.ParticleSetType.mktype("STARS")
        == data.ParticleSetType.mktype(0)
        == data.ParticleSetType.STARS
    ) and data.ParticleSetType.STARS.humanize() == "stars"

    assert (
        data.ParticleSetType.mktype("dark_matter")
        == data.ParticleSetType.mktype("DARK_MATTER")
        == data.ParticleSetType.mktype(1)
        == data.ParticleSetType.DARK_MATTER
    ) and data.ParticleSetType.DARK_MATTER.humanize() == "dark_matter"

    assert (
        data.ParticleSetType.mktype("gas")
        == data.ParticleSetType.mktype("GAS")
        == data.ParticleSetType.mktype(2)
        == data.ParticleSetType.GAS
    ) and data.ParticleSetType.GAS.humanize() == "gas"

    with pytest.raises(ValueError):
        data.ParticleSetType.mktype(43)
    with pytest.raises(ValueError):
        data.ParticleSetType.mktype("foo")
    with pytest.raises(ValueError):
        data.ParticleSetType.mktype(None)


# =============================================================================
# PARTICLE_SET TESTS
# =============================================================================


def test_ParticleSet_creation_with_potential(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=True
    )
    pset = data.ParticleSet(
        data.ParticleSetType.STARS,
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

    assert pset.ptype == data.ParticleSetType.STARS
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
    pset = data.ParticleSet(
        data.ParticleSetType.STARS,
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

    assert pset.ptype == data.ParticleSetType.STARS
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
        data.ParticleSet(data.ParticleSetType.STARS, **params)


def test_ParticleSet_len(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=True
    )

    pset = data.ParticleSet(
        data.ParticleSetType.STARS,
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

    pset = data.ParticleSet(
        data.ParticleSetType.STARS,
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
            "ptype": data.ParticleSetType.STARS.humanize(),
            "ptypev": data.ParticleSetType.STARS.value,
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
            "total_energy": (
                0.5 * (vx ** 2 + vy ** 2 + vz ** 2) + pot
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


@pytest.mark.parametrize("has_potential", [True, False])
def test_ParticleSet_repr(data_particleset, has_potential):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(
        seed=42, has_potential=has_potential
    )

    pset = data.ParticleSet(
        data.ParticleSetType.STARS,
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
        f"ParticleSet(STARS, size={len(m)}, "
        f"softening={soft}, potentials={has_potential})"
    )

    assert repr(pset) == expected


def test_ParticleSet_angular_momentum(data_particleset):
    m, x, y, z, vx, vy, vz, soft, pot = data_particleset(seed=42)

    pset = data.ParticleSet(
        data.ParticleSetType.STARS,
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


# =============================================================================
# TEST GALAXY MANUAL
# =============================================================================


def test_Galaxy_invalid_ParticleSetType(data_particleset):

    bad_types = {
        "stars": data.ParticleSetType.DARK_MATTER,  # WRONG
        "dark_matter": data.ParticleSetType.GAS,
        "gas": data.ParticleSetType.STARS,
    }

    gal_kwargs = {}
    for pname, ptype in bad_types.items():
        m, x, y, z, vx, vy, vz, soft, pot = data_particleset()
        pset = data.ParticleSet(
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
        data.Galaxy(**gal_kwargs)


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

    gal = data.mkgalaxy(
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
        data.mkgalaxy(**params)


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

    gal = data.mkgalaxy(
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

    gkwargs = data.galaxy_as_kwargs(gal)

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
# PLOTTER
# =============================================================================


def test_Galaxy_kinectic_plot(galaxy):
    gal = galaxy()
    assert isinstance(gal.plot, plot.GalaxyPlotter)
    assert gal.plot._galaxy is gal


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
