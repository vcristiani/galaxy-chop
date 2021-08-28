# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Utilities module."""

# =============================================================================
# IMPORTS
# =============================================================================


# =============================================================================
# BACKENDS
# =============================================================================


def center(
    x,
    y,
    z,
    potential,
):
    """Centers the particles.

    Centers the position of all particles in the galaxy respect
    to the position of the lowest potential dark matter particle.

    Parameters
    ----------
    x_s, y_s, z_s : `np.ndarray(n_s,1)`
        Star positions.
    x_dm, y_dm, z_dm : `np.ndarray(n_dm,1)`
        Dark matter positions.
    x_g, y_g, z_g : `np.ndarray(n_g,1)`
        Gas positions.

    Returns
    -------
    tuple : `np.ndarray`
        x_s : `np.ndarray(n_s,1)`
            Centered star positions.
        y_s : `np.ndarray(n_s,1)`
            Centered star positions.
        z_s : `np.ndarray(n_s,1)`
            Centered star positions.
        x_dm : `np.ndarray(n_dm,1)`
            Centered dark matter positions.
        y_dm : `np.ndarray(n_dm,1)`
            Centered dark matter positions.
        z_dm : `np.ndarray(n_dm,1)`
            Centered dark matter positions.
        x_g : `np.ndarray(n_g,1)`
            Centered gas positions.
        y_g : `np.ndarray(n_g,1)`
            Centered gas positions.
        z_g : `np.ndarray(n_g,1)`
            Centered gas positions.

    """

    argmin = potential[1].argmin()

    x_s = x[0] - x[1][argmin]
    y_s = y[0] - y[1][argmin]
    z_s = z[0] - z[1][argmin]

    x_dm = x[1] - x[1][argmin]
    y_dm = y[1] - y[1][argmin]
    z_dm = z[1] - z[1][argmin]

    x_g = x[2] - x[1][argmin]
    y_g = y[2] - y[1][argmin]
    z_g = z[2] - z[1][argmin]

    return x_s, y_s, z_s, x_dm, y_dm, z_dm, x_g, y_g, z_g
