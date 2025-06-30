# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Test utilities  galaxychop.preproc.salign"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings

from galaxychop.preproc import salign

import pandas as pd

import pytest


# =============================================================================
# ALIGN
# =============================================================================


@pytest.mark.filterwarnings("ignore:star_align")
def test_star_align_rcur0dot9(galaxy):
    gal = galaxy(seed=42)

    agal = salign.star_align(gal, r_cut=0.9)

    # Catch the warning:
    with pytest.warns(UserWarning):
        warnings.warn(
            "Input Galaxy is not centered. Please, center it \
            with Centralizer.transform(galaxy, with_potential) \
            or proceed with caution.",
            UserWarning,
        )

    df = gal.to_dataframe()
    adf = agal.to_dataframe()

    changed = [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "Jx",
        "Jy",
        "Jz",
        "kinetic_energy",
        "total_energy",
    ]

    for colname in df.columns[~df.columns.isin(changed)]:
        ocol = df[colname]
        acol = adf[colname]
        assert (ocol == acol).all(), colname

    for colname in changed:
        ocol = df[colname]
        acol = adf[colname]
        assert not (ocol == acol).all(), colname


@pytest.mark.filterwarnings("ignore:star_align")
def test_star_align(galaxy):
    gal = galaxy(seed=42)

    agal = salign.star_align(gal)

    # Catch the warning:
    with pytest.warns(UserWarning):
        warnings.warn(
            "Input Galaxy is not centered. Please, center it \
            with Centralizer.transform(galaxy, with_potential) \
            or proceed with caution.",
            UserWarning,
        )

    df = gal.to_dataframe()
    adf = agal.to_dataframe()

    changed = [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "Jx",
        "Jy",
        "Jz",
        "kinetic_energy",
        "total_energy",
    ]

    for colname in df.columns[~df.columns.isin(changed)]:
        ocol = df[colname]
        acol = adf[colname]
        assert (ocol == acol).all(), colname

    for colname in changed:
        ocol = df[colname]
        acol = adf[colname]
        assert not (ocol == acol).all(), colname


def test_star_align_invalid_rcut(galaxy):
    gal = galaxy(seed=42)

    with pytest.raises(ValueError):
        salign.star_align(gal, r_cut=-1)


@pytest.mark.filterwarnings("ignore:star_align")
def test_is_star_aligned_real_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    agal = salign.star_align(gal, r_cut=5)

    assert not salign.is_star_aligned(gal, r_cut=5)
    assert salign.is_star_aligned(agal, r_cut=5)


@pytest.mark.filterwarnings("ignore:star_align")
def test_aligner_transformer(galaxy):
    gal = galaxy(seed=42)
    aligner = salign.Aligner()

    class_agal = aligner.transform(gal)
    class_df = class_agal.to_dataframe()

    func_agal = salign.star_align(gal)
    func_df = func_agal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)


@pytest.mark.filterwarnings("ignore:star_align")
def test_aligner_default_r_cut(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    aligner = salign.Aligner()

    class_agal = aligner.transform(gal)
    class_df = class_agal.to_dataframe()

    func_agal = salign.star_align(gal, r_cut=30)
    func_df = func_agal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)


@pytest.mark.filterwarnings("ignore:star_align")
def test_aligner_notdefault_r_cut(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    aligner = salign.Aligner(r_cut=10)

    class_agal = aligner.transform(gal)
    class_df = class_agal.to_dataframe()

    func_agal = salign.star_align(gal, r_cut=10)
    func_df = func_agal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)


@pytest.mark.filterwarnings("ignore:star_align")
def test_aligner_checker(galaxy):
    gal = galaxy(seed=42)
    aligner = salign.Aligner()
    agal = aligner.transform(gal)

    assert (aligner.checker(gal)) == (salign.is_star_aligned(gal))
    assert (aligner.checker(agal)) == (salign.is_star_aligned(agal))
