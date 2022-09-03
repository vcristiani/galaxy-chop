# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities  galaxychop.preproc.potential_energy"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import data
from galaxychop.preproc import potential_energy

import numpy as np
import numpy.testing as npt

import pytest


# =============================================================================
#   POTENTIAL ENERGY
# =============================================================================


def test_Galaxy_potential_energy_already_calculated(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=True,
        dm_potential=True,
        gas_potential=True,
    )
    with pytest.raises(ValueError):
        potential_energy.potential(gal)


def test_Galaxy_potential_energy(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal = potential_energy.potential(gal)

    assert isinstance(pgal, data.Galaxy)
    assert np.all(pgal.stars.potential == pgal.potential_energy_[0])
    assert np.all(pgal.dark_matter.potential == pgal.potential_energy_[1])
    assert np.all(pgal.gas.potential == pgal.potential_energy_[2])


@pytest.mark.skipif(
    potential_energy.DEFAULT_POTENTIAL_BACKEND == "numpy",
    reason="apparently the potential fortran extension are not compiled",
)
def test_Galaxy_potential_energy_fortran_backend(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal_f = potential_energy.potential(gal, backend="fortran")

    assert isinstance(pgal_f, data.Galaxy)
    assert np.all(pgal_f.stars.potential == pgal_f.potential_energy_[0])
    assert np.all(pgal_f.dark_matter.potential == pgal_f.potential_energy_[1])
    assert np.all(pgal_f.gas.potential == pgal_f.potential_energy_[2])


@pytest.mark.skipif(
    potential_energy.DEFAULT_POTENTIAL_BACKEND == "numpy",
    reason="apparently the potential fortran extension are not compiled",
)
@pytest.mark.slow
def test_Galaxy_potential_energy_backend_consistency(galaxy):
    gal = galaxy(
        seed=42,
        stars_potential=False,
        dm_potential=False,
        gas_potential=False,
    )

    pgal_np = potential_energy.potential(gal, backend="numpy")
    pgal_f = potential_energy.potential(gal, backend="fortran")

    decimal = 5
    npt.assert_almost_equal(
        pgal_np.stars.potential.value, pgal_f.stars.potential.value, decimal
    )
    npt.assert_almost_equal(
        pgal_np.dark_matter.potential.value,
        pgal_f.dark_matter.potential.value,
        decimal,
    )
    npt.assert_almost_equal(
        pgal_np.gas.potential.value, pgal_f.gas.potential.value, decimal
    )


@pytest.mark.xfail
@pytest.mark.skipif(
    potential_energy.DEFAULT_POTENTIAL_BACKEND == "numpy",
    reason="apparently the potential fortran extension are not compiled",
)
@pytest.mark.slow
def test_potential_recover(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    kwargs = {
        k: v
        for k, v in data.galaxy_as_kwargs(gal).items()
        if "potential_" not in k
    }
    new = potential_energy.potential(
        data.mkgalaxy(**kwargs), backend="fortran"
    )

    original_potential = (
        gal.to_dataframe(attributes=["potential"]).to_numpy().flatten()
    )

    new_potential = (
        new.to_dataframe(attributes=["potential"]).to_numpy().flatten()
    )

    np.testing.assert_allclose(original_potential, new_potential)
