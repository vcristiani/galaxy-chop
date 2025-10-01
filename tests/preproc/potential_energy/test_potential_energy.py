# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================


"""Test utilities  galaxychop.preproc.potential_energy"""


# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop.core.galaxy import Galaxy
from galaxychop.preproc import potential_energy

import numpy as np
import numpy.testing as npt

import pandas as pd

import pytest


# =============================================================================
# POTENTIAL ENERGY
# =============================================================================


def test_Galaxy_potential_energy_already_calculated(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )

    potential_energy.potential(gal)

    with (pytest.warns(
        UserWarning,
        match="Galaxy potential is already calculated"
    )):
        potential_energy.potential(gal)


def test_Galaxy_potential_energy(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal = potential_energy.potential(gal)

    assert isinstance(pgal, Galaxy)
    assert np.all(pgal.stars.potential == pgal.potential_energy_[0])
    assert np.all(pgal.dark_matter.potential == pgal.potential_energy_[1])
    assert np.all(pgal.gas.potential == pgal.potential_energy_[2])


def test_Galaxy_potential_energy_numba_backend(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal_b = potential_energy.potential(gal, backend="numba")

    assert isinstance(pgal_b, Galaxy)
    assert np.all(pgal_b.stars.potential == pgal_b.potential_energy_[0])
    assert np.all(pgal_b.dark_matter.potential == pgal_b.potential_energy_[1])
    assert np.all(pgal_b.gas.potential == pgal_b.potential_energy_[2])


@pytest.mark.slow
def test_Galaxy_potential_energy_backend_consistency(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal_np = potential_energy.potential(gal, backend="numpy")
    pgal_b = potential_energy.potential(gal, backend="numba")

    decimal = 5
    npt.assert_almost_equal(
        pgal_np.stars.potential.value, pgal_b.stars.potential.value, decimal
    )
    npt.assert_almost_equal(
        pgal_np.dark_matter.potential.value,
        pgal_b.dark_matter.potential.value,
        decimal,
    )
    npt.assert_almost_equal(
        pgal_np.gas.potential.value, pgal_b.gas.potential.value, decimal
    )


@pytest.mark.slow
def test_Galaxy_potential_energy_backend_consistency_grispy(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal_gsp = potential_energy.potential(gal, backend="grispy")
    pgal_b = potential_energy.potential(gal, backend="numba")

    decimal = 2
    npt.assert_almost_equal(
        pgal_gsp.stars.potential.value, pgal_b.stars.potential.value, decimal
    )
    npt.assert_almost_equal(
        pgal_gsp.dark_matter.potential.value,
        pgal_b.dark_matter.potential.value,
        decimal,
    )
    npt.assert_almost_equal(
        pgal_gsp.gas.potential.value, pgal_b.gas.potential.value, decimal
    )


@pytest.mark.xfail
@pytest.mark.slow
def test_potential_recover(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    kwargs = {
        k: v for k, v in gal.disassemble().items() if "potential_" not in k
    }
    new = potential_energy.potential(
        Galaxy.mkgalaxy(**kwargs), backend="numba"
    )

    original_potential = (
        gal.to_dataframe(attributes=["potential"]).to_numpy().flatten()
    )

    new_potential = (
        new.to_dataframe(attributes=["potential"]).to_numpy().flatten()
    )

    np.testing.assert_allclose(original_potential, new_potential)


@pytest.mark.slow
def test_potentializer_transformer(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )

    potentializer = potential_energy.Potentializer("numba")

    class_pgal = potentializer.transform(gal)
    class_df = class_pgal.to_dataframe()

    func_pgal = potential_energy.potential(gal, backend="numba")
    func_df = func_pgal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)
