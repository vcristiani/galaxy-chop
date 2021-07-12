# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Fixtures input data."""

# =============================================================================
# IMPORTS
# =============================================================================

import functools
import os
from pathlib import Path

from galaxychop import core

import numpy as np

import pytest

# =============================================================================
# PATHS
# =============================================================================

PATH = Path(os.path.abspath(os.path.dirname(__file__)))

TEST_DATA_PATH = PATH / "test_data"

TEST_DATA_REAL_PATH = TEST_DATA_PATH / "real"

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def data_galaxy():
    def make(
        seed=None,
        stars=100,
        dm=200,
        gas=300,
        stars_number_min=10_000,
        stars_number_max=20_000,
        dm_number_min_min=10_000,
        dm_number_max=20_000,
        gas_number_min=10_000,
        gas_number_max=20_000
        # stars_softening_min=0., stars_softening_max=.5,
        # dm_softening_min_min=0., dm_softening_max=.5,
        # gas_softening_min=0., gas_softening_max=.5,
        # potential_stars=True,
        # potential_gas=True,
        # potential_dm=True
    ):

        # start the random generator
        random = np.random.default_rng(seed=seed)

        x_s = random.random(stars)
        y_s = random.random(stars)
        z_s = random.random(stars)
        vx_s = random.random(stars)
        vy_s = random.random(stars)
        vz_s = random.random(stars)
        m_s = random.random(stars)

        x_dm = random.random(dm)
        y_dm = random.random(dm)
        z_dm = random.random(dm)
        vx_dm = random.random(dm)
        vy_dm = random.random(dm)
        vz_dm = random.random(dm)
        m_dm = random.random(dm)

        x_g = random.random(gas)
        y_g = random.random(gas)
        z_g = random.random(gas)
        vx_g = random.random(gas)
        vy_g = random.random(gas)
        vz_g = random.random(gas)
        m_g = random.random(gas)

        softening_s = 0.0
        softening_dm = 0.0
        softening_g = 0.0
        pot_s = random.random(stars)
        pot_dm = random.random(dm)
        pot_g = random.random(gas)

        return (
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
        )

    return make


@pytest.fixture(scope="session")
def galaxy(data_galaxy):
    @functools.wraps(data_galaxy)
    def make(*args, **kwargs):
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
        ) = data_galaxy(*args, **kwargs)

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
        return gal

    return make
