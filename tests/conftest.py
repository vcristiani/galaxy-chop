# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Fixtures input data."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path

from galaxychop import data, io

import numpy as np

import pytest

# =============================================================================
# PATHS
# =============================================================================

PATH = Path(os.path.abspath(os.path.dirname(__file__)))

TEST_DATA_PATH = PATH / "datasets"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def data_path():
    return TEST_DATA_PATH.joinpath


@pytest.fixture(scope="session")
def read_hdf5_galaxy(data_path):
    def read(filename):
        path = data_path(filename)
        return io.read_hdf5(path)

    return read


@pytest.fixture(scope="session")
def data_particleset():
    def make(
        *,
        seed=None,
        size_min=100,
        size_max=1000,
        soft_min=0.0,
        soft_max=1.0,
        has_potential=True,
    ):
        random = np.random.default_rng(seed=seed)

        size = random.integers(size_min, size_max, endpoint=True)

        m = random.random(size)
        x = random.random(size)
        y = random.random(size)
        z = random.random(size)
        vx = random.random(size)
        vy = random.random(size)
        vz = random.random(size)
        soft = random.uniform(soft_min, soft_max)
        pot = random.random(size) if has_potential else None

        return m, x, y, z, vx, vy, vz, soft, pot

    return make


@pytest.fixture(scope="session")
def data_galaxy(data_particleset):
    def make(
        *,
        seed=None,
        stars_min=100,
        stars_max=100,
        stars_softening_min=0.0,
        stars_softening_max=1.0,
        stars_potential=True,
        dm_min=100,
        dm_max=1000,
        dm_softening_min=0.0,
        dm_softening_max=1.0,
        dm_potential=True,
        gas_min=100,
        gas_max=1000,
        gas_softening_min=0.0,
        gas_softening_max=1.0,
        gas_potential=True,
    ):
        # start the random generator
        random = np.random.default_rng(seed=seed)

        # STARS
        stars_data = data_particleset(
            seed=random,
            size_min=stars_min,
            size_max=stars_max,
            soft_min=stars_softening_min,
            soft_max=stars_softening_max,
            has_potential=stars_potential,
        )

        # DARK_MATTER
        dm_data = data_particleset(
            seed=random,
            size_min=dm_min,
            size_max=dm_max,
            soft_min=dm_softening_min,
            soft_max=dm_softening_max,
            has_potential=dm_potential,
        )

        # GAS
        gas_data = data_particleset(
            seed=random,
            size_min=gas_min,
            size_max=gas_max,
            soft_min=gas_softening_min,
            soft_max=gas_softening_max,
            has_potential=gas_potential,
        )

        return stars_data + dm_data + gas_data

    return make


@pytest.fixture(scope="session")
def galaxy(data_galaxy):
    # @functools.wraps(data_galaxy)
    def make(**kwargs):
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
        ) = data_galaxy(**kwargs)

        gal = data.mkgalaxy(
            # stars
            m_s=m_s,
            x_s=x_s,
            y_s=y_s,
            z_s=z_s,
            vx_s=vx_s,
            vy_s=vy_s,
            vz_s=vz_s,
            softening_s=soft_s,
            potential_s=potential_s,
            # dark matter
            m_dm=m_dm,
            x_dm=x_dm,
            y_dm=y_dm,
            z_dm=z_dm,
            vx_dm=vx_dm,
            vy_dm=vy_dm,
            vz_dm=vz_dm,
            softening_dm=soft_dm,
            potential_dm=potential_dm,
            # gas
            m_g=m_g,
            x_g=x_g,
            y_g=y_g,
            z_g=z_g,
            vx_g=vx_g,
            vy_g=vy_g,
            vz_g=vz_g,
            softening_g=soft_g,
            potential_g=potential_g,
        )
        return gal

    return make
