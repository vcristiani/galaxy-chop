# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for crop a galaxy based on stellar mass."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ..core import data

# =============================================================================
# API
# =============================================================================


def _get_star_cut_indexes(sdf, cut_radius_factor):
    sdf = sdf[["x", "y", "z", "m"]].copy()

    sdf["radius"] = np.sqrt(sdf.x**2 + sdf.y**2 + sdf.z**2)
    sdf.drop(["x", "y", "z"], axis="columns", inplace=True)

    sdf.sort_values("radius", inplace=True)

    sdf["m_cumsum"] = sdf.m.cumsum()
    sdf.drop(["m"], axis="columns", inplace=True)

    half_m_cumsum = sdf.iloc[-1].m_cumsum / 2
    sdf["half_m_cumsum_diff"] = np.abs(sdf.m_cumsum - half_m_cumsum)

    cut_radius = (
        sdf.iloc[sdf.half_m_cumsum_diff.argmin()].radius * cut_radius_factor
    )

    cut_idxs = sdf[sdf.radius > cut_radius].index.to_numpy()

    del sdf

    return cut_idxs, cut_radius


def star_mass_radius_crop(galaxy, *, num_radii=3):
    """Crop select stars within a specified number of the radii enclosing \
    half fractions of the stellar mass.

    Parameters
    ----------
    galaxy : galaxychop.Galaxy
        The galaxy object for which to determine half of the
        mass-enclosing radii.
    num_radii : int, optional
        The number of radii to consider. Default is 3.

    Returns
    -------
    galaxychop.Galaxy
        A new galaxy object containing stars within the specified radii
        enclosing various fractions of the stellar mass.

    """

    # We convert the stars into a dataframe
    stars_df = galaxy.stars.to_dataframe()

    # We check which rows to delete and what cutoff radius it gives us
    to_trim_idxs, _ = _get_star_cut_indexes(
        stars_df, cut_radius_factor=num_radii
    )
    trim_stars_df = stars_df.drop(to_trim_idxs, axis="rows")

    # We create a new particle set with the new stars.
    trim_stars = data.ParticleSet(
        ptype=data.ParticleSetType.STARS,
        m=trim_stars_df["m"].values,
        x=trim_stars_df["x"].values,
        y=trim_stars_df["y"].values,
        z=trim_stars_df["z"].values,
        vx=trim_stars_df["vx"].values,
        vy=trim_stars_df["vy"].values,
        vz=trim_stars_df["vz"].values,
        potential=trim_stars_df["potential"].values,
        softening=galaxy.stars.softening,
    )

    del trim_stars_df

    dm = galaxy.dark_matter.copy()
    gas = galaxy.gas.copy()

    trim_galaxy = data.Galaxy(stars=trim_stars, dark_matter=dm, gas=gas)

    return trim_galaxy
