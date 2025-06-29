# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Test utilities  galaxychop.preproc.smr_crop"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop.preproc import smr_crop

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_half_star_mass_radius_crop(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    cgal = smr_crop.half_star_mass_radius_crop(gal, num_radii=1)

    gal_m_s = gal.stars.m.sum().to_value()
    cgal_m_s = cgal.stars.m.sum().to_value()

    assert np.isclose(
        cgal_m_s, (gal_m_s / 2.0), rtol=1e-05, atol=1e-08, equal_nan=False
    )


def test_gal_crop_invalid_num_radii(galaxy):
    gal = galaxy(seed=42)

    with pytest.raises(ValueError):
        smr_crop.half_star_mass_radius_crop(gal, num_radii=-1)


def test_is_gal_cutted_real_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    cgal = smr_crop.half_star_mass_radius_crop(gal)

    # Rev "is_star_cutted" porque mepa que vuelve a
    # cortar la glx...
    # assert not smr_crop.is_star_cutted(gal, num_radii=1)
    assert smr_crop.is_star_cutted(cgal, num_radii=1)


def test_cutter_transformer(galaxy):
    gal = galaxy(seed=42)
    cutter = smr_crop.Cutter(num_radii=1)

    class_cgal = cutter.transform(gal)
    class_df = class_cgal.to_dataframe()

    func_cgal = smr_crop.half_star_mass_radius_crop(gal, num_radii=1)
    func_df = func_cgal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)


def test_cutter_default_num_radii(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    cutter = smr_crop.Cutter()

    class_cgal = cutter.transform(gal)
    class_df = class_cgal.to_dataframe()

    func_cgal = smr_crop.half_star_mass_radius_crop(gal, num_radii=3)
    func_df = func_cgal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)


def test_cutter_checker(galaxy):
    gal = galaxy(seed=42)
    cutter = smr_crop.Cutter()
    cgal = cutter.transform(gal)

    assert (cutter.checker(gal)) == (smr_crop.is_star_cutted(gal))
    assert (cutter.checker(cgal)) == (smr_crop.is_star_cutted(cgal))


@pytest.mark.parametrize("particle", ["stars", "gas", "dark_matter", "all"])
def test_get_radius_half_mass(read_hdf5_galaxy, particle):
    gal = read_hdf5_galaxy("gal394242.h5")

    half_mass_radius = smr_crop.get_radius_half_mass(gal, particle)

    # Bruno:
    # Lo que debería hacer sería pre-calcular estos valores
    # aparte y exigir que sean lo mismo que a mano (que el
    # "expected_radius"), como para tener control total de
    # lo que está haciendo la función.
    # Como WIP, lo único que voy a pedir es que el radio
    # de cualquier tipo sea > 0, y después otro test para
    # un bad input (catcheando el error de ParticleSetType)
    assert half_mass_radius > 0.0


def test_get_radius_half_mass_badinput(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    with pytest.raises(ValueError):
        smr_crop.get_radius_half_mass(gal, "zaraza")
