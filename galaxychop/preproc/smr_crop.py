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

from ._base import GalaxyTransformerABC
from ..core import data
from ..utils import doc_inherit

# =============================================================================
# INTERNALS
# =============================================================================


def _get_half_smr_crop(sdf, cut_radius_factor):
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


# =============================================================================
# CUTTER CLASS
# =============================================================================


class Cutter(GalaxyTransformerABC):
    """
    Cutter class.

    Given the positions and mass of particles, compute the stellar
    half mass radius (radii of the sphere centered at the origin that
    encloses half of the total sum of stellar particles mass) and
    return only the stellar component inside of a multiple of it.

    """

    def __init__(self, num_radii=3):
        self.num_radii = num_radii

    @doc_inherit(GalaxyTransformerABC.transform)
    def transform(self, galaxy):
        return half_star_mass_radius_crop(galaxy, num_radii=self.num_radii)

    @doc_inherit(GalaxyTransformerABC.checker)
    def checker(self, galaxy, **kwargs):
        return is_star_cutted(galaxy, num_radii=self.num_radii, **kwargs)


# =============================================================================
# API FUNCTIONS
# =============================================================================


def half_star_mass_radius_crop(galaxy, *, num_radii=3):
    """
    Crop select stars within a specified number of the radii enclosing \
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
    if num_radii is not None and num_radii <= 0.0:
        raise ValueError("num_radii must not be lower than 0.")

    # We convert the stars into a dataframe
    stars_df = galaxy.stars.to_dataframe()

    # We check which rows to delete and what cutoff radius it gives us
    to_trim_idxs, _ = _get_half_smr_crop(stars_df, cut_radius_factor=num_radii)
    trim_stars_df = stars_df.drop(to_trim_idxs, axis="rows")

    # We create a new particle set with the new stars.
    trim_stars = data.ParticleSet(
        ptype=data.ParticleSetType.STARS,
        m=trim_stars_df["m"].to_numpy(),
        x=trim_stars_df["x"].to_numpy(),
        y=trim_stars_df["y"].to_numpy(),
        z=trim_stars_df["z"].to_numpy(),
        vx=trim_stars_df["vx"].to_numpy(),
        vy=trim_stars_df["vy"].to_numpy(),
        vz=trim_stars_df["vz"].to_numpy(),
        potential=trim_stars_df["potential"].to_numpy(),
        softening=galaxy.stars.softening.value,
    )

    del trim_stars_df

    dm = galaxy.dark_matter.copy()
    gas = galaxy.gas.copy()

    trim_galaxy = data.Galaxy(stars=trim_stars, dark_matter=dm, gas=gas)

    return trim_galaxy


def is_star_cutted(galaxy, *, num_radii=3, rtol=1e-05, atol=1e-08):
    """
    Validate if the galaxy is cutted.

    Parameters
    ----------
    galaxy : ``Galaxy class`` object
    num_radii : float, optional
        Default value =  3. If it's provided,
        it must be positive and the galaxy is
        cutted at that radii, with only particles
        at radii smaller than num_radii*r_half.
    rtol : float
        Relative tolerance. Default value = 1e-05.
    atol : float
        Absolute tolerance. Default value = 1e-08.

    Returns
    -------
    bool
        True if galaxy is cutted at given radii, and
        there is no stellar particles at further distances,
        False otherwise.

    """
    # Now we extract only the needed column to rotate the galaxy
    stars_df = galaxy.stars.to_dataframe(attributes=["m", "x", "y", "z"])

    to_trim_idxs, cut_radius = _get_half_smr_crop(stars_df, num_radii)
    trim_stars_df = stars_df.drop(to_trim_idxs, axis="rows")

    # Distances of all stellar particles that "survives"
    distances = np.sqrt(
        trim_stars_df.x**2 + trim_stars_df.y**2 + trim_stars_df.z**2
    )

    # maximum distance index of all particles
    maxdist_idx = np.argmin(distances)
    max_values = distances[maxdist_idx]

    return np.all(np.less_equal(max_values, cut_radius))


def get_radius_half_mass(galaxy, particle="stars"):
    """Compute the radii.

    that encloses half of the mass of the
    selected type of particle.
    For the complete galaxy, set particle = 'all' (also
    admitted False or '').

    Parameters
    ----------
    galaxy : ``Galaxy class`` object.
        The galaxy object for which to determine half of the
        mass-enclosing radii.
    particle : str, default="star"
        ParticleSetType instance. Defines the type of
        particle to determine a half mass radii e.g.
        'stars', 'gas', 'DM' or 'all'.

    Returns
    -------
    r_half_star : float
        Radii of the sphere that encloses half of the mass for
        the selected type of particles.

    """
    if particle in ["", "all", None, False]:
        # We convert the particles into a dataframe
        df = galaxy.to_dataframe()
    else:
        particle_type = data.ParticleSetType.mktype(particle)
        particle_type = data.ParticleSetType.humanize(particle_type)
        if particle_type == "stars":
            df = galaxy.stars.to_dataframe()
        elif particle_type == "dark_matter":
            df = galaxy.dark_matter.to_dataframe()
        elif particle_type == "gas":
            df = galaxy.gas.to_dataframe()

    # cut_radius_factor = 1 to get the "half mass radius"
    _, r_half = _get_half_smr_crop(df, cut_radius_factor=1)

    return r_half
