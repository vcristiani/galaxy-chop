# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
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

from .. import data

# =============================================================================
# BACKENDS
# =============================================================================


def center(galaxy):
    """
    Particle centring.

    Centers the position of all particle galaxies respect
    to the position of the lowest potential particle.

    Parameters
    ----------
    galaxy : object of Galaxy class.

    Returns
    -------
    galaxy: new object of Galaxy class.
        A new galaxy object with centered positions respect
        to the position of the lowest potential particle.
    """
    if not galaxy.has_potential_:
        raise ValueError("galaxy must has the potential energy")

    # We extract only the needed column to centrer the galaxy
    df = galaxy.to_dataframe(attributes=["ptypev", "x", "y", "z", "potential"])

    # minimum potential index of all particles and we extract data frame row
    minpot_idx = df.potential.argmin()
    min_values = df.iloc[minpot_idx]

    # We subtract all position columns by the position with the lowest
    # potential value and replace this new position columns on dataframe
    columns = ["x", "y", "z"]
    df[columns] = df[columns] - min_values[columns]

    # We split the dataframe by particle type.
    stars = df[df.ptypev == data.ParticleSetType.STARS.value]
    dark_matter = df[df.ptypev == data.ParticleSetType.DARK_MATTER.value]
    gas = df[df.ptypev == data.ParticleSetType.GAS.value]

    # patch
    new = data.galaxy_as_kwargs(galaxy)

    new.update(
        x_s=stars.x.to_numpy(),
        y_s=stars.y.to_numpy(),
        z_s=stars.z.to_numpy(),
        x_dm=dark_matter.x.to_numpy(),
        y_dm=dark_matter.y.to_numpy(),
        z_dm=dark_matter.z.to_numpy(),
        x_g=gas.x.to_numpy(),
        y_g=gas.y.to_numpy(),
        z_g=gas.z.to_numpy(),
    )

    return data.mkgalaxy(**new)


def is_centered(galaxy, rtol=1e-05, atol=1e-08):
    """
    Validate if the galaxy is centered.

    Parameters
    ----------
    galaxy : object of Galaxy class.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.

    Returns
    -------
    bool
        True if galaxy is centered respect to the position of the lowest
        potential particle, False otherwise.
    """
    if not galaxy.has_potential_:
        raise ValueError("galaxy must has the potential energy")

    # We extract only the needed column to centrer the galaxy
    df = galaxy.to_dataframe(attributes=["x", "y", "z", "potential"])

    # minimum potential index of all particles and we extract data frame row
    minpot_idx = df.potential.argmin()
    min_values = df.iloc[minpot_idx]

    return np.allclose(min_values[["x", "y", "z"]], 0, rtol=rtol, atol=atol)
