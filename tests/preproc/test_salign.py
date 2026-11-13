# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2020, 2021, 2022, 2023, 2024, 2025, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Test utilities  galaxychop.preproc.salign"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop.preproc import salign

import pandas as pd

import pytest

# =============================================================================
# ALIGN
# =============================================================================


def test_star_align_rcur0dot9(galaxy):
    gal = galaxy(seed=42)

    with pytest.warns(UserWarning, match="not centered"):
        agal = salign.star_align(gal, r_cut=0.9)

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
        "radius",
        "ptype_circular_velocity",
        "galaxy_circular_velocity",
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


def test_star_align(galaxy):
    gal = galaxy(seed=42)

    with pytest.warns(UserWarning, match="not centered"):
        agal = salign.star_align(gal)

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
        "radius",
        "ptype_circular_velocity",
        "galaxy_circular_velocity",
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


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_star_align_invalid_rcut(galaxy):
    gal = galaxy(seed=42)

    with pytest.raises(ValueError):
        salign.star_align(gal, r_cut=-1)


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_is_star_aligned_real_galaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    agal = salign.star_align(gal, r_cut=5)

    assert not salign.is_star_aligned(gal, r_cut=5)
    assert salign.is_star_aligned(agal, r_cut=5)


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_aligner_transformer(galaxy):
    gal = galaxy(seed=42)
    aligner = salign.Aligner()

    class_agal = aligner.transform(gal)
    class_df = class_agal.to_dataframe()

    func_agal = salign.star_align(gal)
    func_df = func_agal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_aligner_default_r_cut(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    aligner = salign.Aligner()

    class_agal = aligner.transform(gal)
    class_df = class_agal.to_dataframe()

    func_agal = salign.star_align(gal, r_cut=30)
    func_df = func_agal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_aligner_notdefault_r_cut(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    aligner = salign.Aligner(r_cut=10)

    class_agal = aligner.transform(gal)
    class_df = class_agal.to_dataframe()

    func_agal = salign.star_align(gal, r_cut=10)
    func_df = func_agal.to_dataframe()

    pd.testing.assert_frame_equal(class_df, func_df, check_dtype=False)


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_aligner_checker(galaxy):
    gal = galaxy(seed=42)
    aligner = salign.Aligner()
    agal = aligner.transform(gal)

    assert (aligner.checker(gal)) == (salign.is_star_aligned(gal))
    assert (aligner.checker(agal)) == (salign.is_star_aligned(agal))


def test_default_r_cut_is_shared():
    # every alignment entry point must agree on the default radius
    import inspect

    from galaxychop import preproc

    assert salign.DEFAULT_R_CUT == 30
    for func in (
        salign.Aligner,
        salign.star_align,
        salign.is_star_aligned,
        preproc.center_and_align,
        preproc.is_centered_and_aligned,
    ):
        default = inspect.signature(func).parameters["r_cut"].default
        assert default == salign.DEFAULT_R_CUT, func.__name__


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_star_align_default_r_cut(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    default = salign.star_align(gal).to_dataframe()
    with_30 = salign.star_align(gal, r_cut=30).to_dataframe()
    every_star = salign.star_align(gal, r_cut=None).to_dataframe()

    pd.testing.assert_frame_equal(default, with_30)
    assert not default[["x", "y", "z"]].equals(every_star[["x", "y", "z"]])


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_star_align_no_stars_within_r_cut(galaxy):
    # the synthetic stars live in [0, 1), so none is within 1e-3 of the origin
    gal = galaxy(seed=42)

    with pytest.raises(ValueError, match="No stars within r_cut"):
        salign.star_align(gal, r_cut=1e-3)


@pytest.mark.filterwarnings("ignore:Input Galaxy is not centered")
def test_aligner_checker_uses_its_r_cut(read_hdf5_galaxy, monkeypatch):
    gal = read_hdf5_galaxy("gal394242.h5")
    calls = []
    monkeypatch.setattr(
        salign,
        "is_star_aligned",
        lambda galaxy, **kwargs: calls.append(kwargs) or True,
    )

    salign.Aligner(r_cut=10).checker(gal)
    salign.Aligner(r_cut=10).checker(gal, r_cut=5)

    assert calls == [{"r_cut": 10}, {"r_cut": 5}]
