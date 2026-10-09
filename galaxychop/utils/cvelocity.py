# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Circular velocity from the enclosed mass."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .. import constants as const

# =============================================================================
# FUNCTIONS
# =============================================================================


def circular_velocity(mass, radius):
    """
    Circular velocity from the mass enclosed within each radius.

    Sorts ``mass`` by ``radius``, accumulates it, and returns
    sqrt(G * M(<r) / r) for every input element, in the original order.
    Particles at radius 0 get NaN (the enclosed mass there is singular).

    Which particles go in ``mass`` decides what the result means: all the
    particles of a galaxy give its rotation curve, while a subset (a
    particle type, or a component of a decomposition) gives only that
    subset's contribution to it, as if it were the only source of the
    potential.

    Parameters
    ----------
    mass : np.ndarray(n)
        Particle masses, in M_sun.
    radius : np.ndarray(n)
        Distance of each particle to the origin, in kpc.

    Returns
    -------
    np.ndarray(n)
        Circular velocity in km/s, in the same order as the inputs.

    Notes
    -----
    If ``mass`` and ``radius`` have different lengths, this returns NaNs
    instead of raising, so a caller validating lengths elsewhere can raise
    its own, clearer error.
    """
    if len(mass) != len(radius):
        return np.full(len(radius), np.nan)

    order = np.argsort(radius)
    enclosed_mass = np.cumsum(mass[order])
    with np.errstate(divide="ignore", invalid="ignore"):
        vcirc = np.sqrt(const.G * enclosed_mass / radius[order])
    vcirc[radius[order] == 0] = np.nan

    result = np.empty_like(vcirc)
    result[order] = vcirc
    return result
