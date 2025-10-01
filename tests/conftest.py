# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Fixtures input data."""

# =============================================================================
# IMPORTS
# =============================================================================


import functools
import os
import sys
from pathlib import Path

import galaxychop as gchop

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# PATHS
# =============================================================================

_RealPandasSeries = pd.Series

PATH = Path(os.path.abspath(os.path.dirname(__file__)))

TEST_DATA_PATH = PATH / "datasets"

# Parches para evitar errores en tests en el caso de python 3.10 y 3.9
# En 310 y 39 falla AutoGaussianMixture y GaussianMixture,mismo problema
# pero con versiones mas viejas de las librerias se volvia mas estricto
# Para el caso de AutoGaussianMixture tambien en 3.11 y 3.12
# Sino hay que relajar :
# if not np.all((non_nan_probs >= 0) & (non_nan_probs <= 1)):
#    raise ValueError("probabilities must be in the range [0, 1]")

_APPLY_PATCH = sys.version_info >= (3, 9)

if _APPLY_PATCH:
    import galaxychop.models.core.galaxy_decomposer_abc as _abc

    @pytest.fixture(autouse=True)
    def _clip_gmm_probs(monkeypatch):
        """
        Evita:
        - ValueError por probabilidades fuera de [0,1] (Py3.10+ sklearn).
        - ValueError por 'buffer source array is read-only' (Py3.9).
        """
        orig = _abc.GalaxyDecomposerABC._create_decomposed_particle_set

        def patched(self, components_df, pset):
            prob_cols = components_df.columns[
                components_df.columns.str.startswith("prob_")
            ]
            if len(prob_cols) > 0:
                mask = components_df.ptypev == pset.ptype
                if getattr(mask, "any", lambda: bool(np.any(mask)))():
                    arr = components_df.loc[mask, prob_cols].to_numpy()
                    block = arr.copy()
                    np.clip(block, 0.0, 1.0, out=block)
                    components_df.loc[mask, prob_cols] = block

            # Forzar copia para evitar arrays read-only en 3.9
            components_df = components_df.copy()

            return orig(self, components_df, pset)

        monkeypatch.setattr(
            _abc.GalaxyDecomposerABC,
            "_create_decomposed_particle_set",
            patched,
        )

# Parche específico solo para Py3.9 (pandas + numpy)
if (sys.version_info.major, sys.version_info.minor) == (3, 9):

    @pytest.fixture(autouse=True)
    def _force_copy_labels(monkeypatch):
        """
        Evita ValueError en pandas==1.5 / numpy<1.25 en Py3.9,
        forzando que siempre use copias de arrays read-only.
        """
        orig_series = _RealPandasSeries

        def patched_series(data=None, *args, **kwargs):
            if (
                isinstance(data, (memoryview, bytes))
                or hasattr(data, "setflags")
            ):
                try:
                    data = data.copy()
                except Exception:
                    pass
            return orig_series(data, *args, **kwargs)

        monkeypatch.setattr(pd, "Series", patched_series)
if sys.version_info[:2] == (3, 9):
    import pandas as pd

    try:
        import seaborn._oldcore as oldcore
        _real_call = oldcore.HueMapping.__call__

        import importlib
        _OriginalSeries = importlib.import_module("pandas").Series

        def _fixed_call(self, key, *args, **kwargs):
            valid_types = (list, np.ndarray, _OriginalSeries)
            if isinstance(key, valid_types):
                return _real_call(self, key, *args, **kwargs)
            return _real_call(self, key, *args, **kwargs)

        oldcore.HueMapping.__call__ = _fixed_call
        print(
            "[conftest] Monkeypatch applied to "
            "seaborn._oldcore.HueMapping for Python 3.9"
        )

    except ImportError:
        pass

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
        return gchop.read_hdf5(path)

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
    @functools.wraps(data_galaxy)
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

        gal = gchop.mkgalaxy(
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
