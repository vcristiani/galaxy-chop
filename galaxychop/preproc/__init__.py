# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""preprocessing module."""

# =============================================================================
# IMPORTS
# =============================================================================

from .circ import DEFAULT_CBIN, DEFAULT_REASSIGN, JCirc, jcirc
from .pcenter import center, is_centered
from .potential_energy import (
    DEFAULT_POTENTIAL_BACKEND,
    G,
    POTENTIAL_BACKENDS,
    potential,
)
from .salign import is_star_aligned, star_align


__all__ = [
    # circ
    "JCirc",
    "jcirc",
    "DEFAULT_CBIN",
    "DEFAULT_REASSIGN",
    # pcenter
    "center",
    "is_centered",
    # potential
    "potential",
    "POTENTIAL_BACKENDS",
    "DEFAULT_POTENTIAL_BACKEND",
    "G",
    # salign
    "star_align",
    "is_star_aligned",
    # composition
    "center_and_align",
]

# =============================================================================
# FUNCTIONS
# =============================================================================


def center_and_align(galaxy, *, r_cut=None):
    """Sequentially performs centering and alignment.

    ``center_and_align(gal) <==> star_align(center(gal))``

    Parameters
    ----------
    galaxy : ``Galaxy class`` object
    r_cut : float, optional
        Default value =  None. If it's provided, it must be positive and the
        rotation matrix `A` is calculated from the particles with radii smaller
        than r_cut.

    Returns
    -------
    galaxy: new ``Galaxy class`` object
        A new galaxy object with centered positions respect to the position of
        the lowest potential particle and their total angular momentum aligned
        with the z-axis.

    """
    centered = center(galaxy)
    aligned = star_align(centered, r_cut=r_cut)

    return aligned


def is_centered_and_aligned(galaxy, *, r_cut=None, rtol=1e-05, atol=1e-08):
    """Validate if the galaxy is centered and aligned.

    ``is_center_and_align(gal) <==> is_centered(gal) and is_star_aligned(gal)``

    Parameters
    ----------
    galaxy : ``Galaxy class`` object
    r_cut : float, optional
        Default value =  None. If it's provided, it must be positive and the
        rotation matrix `A` is calculated from the particles with radii smaller
        than r_cut.
    rtol : float
        Relative tolerance. Default value = 1e-05.
    atol : float
        Absolute tolerance. Default value = 1e-08.

    Returns
    -------
    bool
        True if galaxy is centered respect to the position of the lowest
        potential particle, and if the total angular momentum of the galaxy
        is aligned with the z-axis, False otherwise.

    """
    return is_centered(galaxy, rtol=rtol, atol=atol) and is_star_aligned(
        galaxy, r_cut=r_cut, rtol=rtol, atol=atol
    )
