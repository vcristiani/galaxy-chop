# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to center a galaxy."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ._base import GalaxyTransformerABC
from ..core import data
from ..utils import doc_inherit

# =============================================================================
# CENTRALIZER CLASS
# =============================================================================


class Centralizer(GalaxyTransformerABC):
    """
    Centralizer class.

    Given the positions and potential energy of particles, check and
    center their positions and velocities relative to the system.

    """

    def __init__(self, with_potential=True):
        self.with_potential = with_potential

    @doc_inherit(GalaxyTransformerABC.transform)
    def transform(self, galaxy):
        return center(galaxy, with_potential=self.with_potential)

    @doc_inherit(GalaxyTransformerABC.checker)
    def checker(self, galaxy, **kwargs):
        return is_centered(galaxy, **kwargs)


# =============================================================================
# API FUNCTIONS
# =============================================================================


def center(galaxy, with_potential=True):
    """
    Galaxy particle centering.

    Centers the position and velocities of all galaxy particles.
    If with_potential is True, it subtracts the position columns of the
    galaxy dataframe by the position with the lowest potential value.
    If with_potential is False, it subtracts the position of the geometric
    center of the galaxy.
    To correct velocities, it computes the mean velocity of all particles.

    Parameters
    ----------
    galaxy : ``Galaxy class`` object
        A galaxy object to correct its position and velocities.
    with_potential : bool, default value = True
        If True, set the particle with lowest potential as the origin
        (requires the potential of all particles). Otherwise,
        correct by the geometric center of all particles.

    Returns
    -------
    galaxy : new ``Galaxy class`` object
        A new galaxy object with centered positions respect to the position of
        the lowest potential particle.

    """
    if with_potential and not galaxy.has_potential_:
        raise ValueError(
            "Galaxy must has the potential energy. Use \
            with_potential = False"
        )

    if with_potential:
        # We extract only the needed column to centrer the galaxy
        df = galaxy.to_dataframe(
            attributes=[
                "ptypev",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "potential",
                "m",
            ]
        )

        cond = df["ptypev"].eq(0)

        # Total stellar mass
        m_star_tot = np.sum(df[cond]["m"].values)

        # minimum potential index of all particles and we extract data
        # frame row
        minpot_idx = df.potential.argmin()
        min_values = df.iloc[minpot_idx]

        # We subtract all position columns by the position with the lowest
        # potential value and replace this new position columns on dataframe
        columns = ["x", "y", "z"]
        df[columns] = df[columns] - min_values[columns]

    else:
        # We use only positions and mass
        df = galaxy.to_dataframe(
            attributes=["ptypev", "x", "y", "z", "vx", "vy", "vz", "m"]
        )

        # Using only stars (account the gas and dark matter particles
        # may not be the best option to center the galaxy)
        cond = df["ptypev"].eq(0)  # df["ptypev"].eq(0) df["ptypev"] == 0

        # Total stellar mass
        m_star_tot = np.sum(df[cond]["m"].values)

        # Compute the center of mass using only stars
        x_cm = (
            np.sum(np.multiply(df[cond]["x"].values, df[cond]["m"].values))
            / m_star_tot
        )
        y_cm = (
            np.sum(np.multiply(df[cond]["y"].values, df[cond]["m"].values))
            / m_star_tot
        )
        z_cm = (
            np.sum(np.multiply(df[cond]["z"].values, df[cond]["m"].values))
            / m_star_tot
        )

        # We subtract all position columns by the new origin
        # and replace on dataframe
        df.loc[:, "x"] -= x_cm
        df.loc[:, "y"] -= y_cm
        df.loc[:, "z"] -= z_cm

    # Compute the velocity of the center of mass
    # of the galaxy within the cosmological box
    vx_cm = (
        np.sum(np.multiply(df[cond]["vx"].values, df[cond]["m"].values))
        / m_star_tot
    )
    vy_cm = (
        np.sum(np.multiply(df[cond]["vy"].values, df[cond]["m"].values))
        / m_star_tot
    )
    vz_cm = (
        np.sum(np.multiply(df[cond]["vz"].values, df[cond]["m"].values))
        / m_star_tot
    )
    # And modify the dataframe
    df.loc[:, "vx"] -= vx_cm
    df.loc[:, "vy"] -= vy_cm
    df.loc[:, "vz"] -= vz_cm

    # We split the dataframe by particle type.
    stars = df[df.ptypev == data.ParticleSetType.STARS.value]
    dark_matter = df[df.ptypev == data.ParticleSetType.DARK_MATTER.value]
    gas = df[df.ptypev == data.ParticleSetType.GAS.value]

    # patch
    new = galaxy.disassemble()

    new.update(
        x_s=stars.x.to_numpy(),
        y_s=stars.y.to_numpy(),
        z_s=stars.z.to_numpy(),
        vx_s=stars.vx.to_numpy(),
        vy_s=stars.vy.to_numpy(),
        vz_s=stars.vz.to_numpy(),
        x_dm=dark_matter.x.to_numpy(),
        y_dm=dark_matter.y.to_numpy(),
        z_dm=dark_matter.z.to_numpy(),
        vx_dm=dark_matter.vx.to_numpy(),
        vy_dm=dark_matter.vy.to_numpy(),
        vz_dm=dark_matter.vz.to_numpy(),
        x_g=gas.x.to_numpy(),
        y_g=gas.y.to_numpy(),
        z_g=gas.z.to_numpy(),
        vx_g=gas.vx.to_numpy(),
        vy_g=gas.vy.to_numpy(),
        vz_g=gas.vz.to_numpy(),
    )

    return data.mkgalaxy(**new)


def is_centered(galaxy, *, rtol=1e-05, atol=1e-08):
    """
    Validate if the galaxy is centered.

    Parameters
    ----------
    galaxy : ``Galaxy class`` object
    rtol : float
        Relative tolerance. Default value = 1e-05.
    atol : float
        Absolute tolerance. Default value = 1e-08.

    Returns
    -------
    bool
        True if galaxy is centered respect to the position of the lowest
        potential particle, False otherwise.

    """
    if not galaxy.has_potential_:
        raise ValueError("Galaxy must have the potential energy.")

    df = galaxy.to_dataframe(
        attributes=["x", "y", "z", "vx", "vy", "vz", "potential"]
    )

    # minimum potential index of all particles and we extract data frame row
    minpot_idx = df.potential.argmin()
    min_values = df.iloc[minpot_idx]

    return np.allclose(min_values[["x", "y", "z"]], 0, rtol=rtol, atol=atol)
