# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
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

import numpy as np

from ._potential import potential

# =============================================================================
# BACKENDS
# =============================================================================


def center(
    m_s,
    x_s,
    y_s,
    z_s,
    m_dm,
    x_dm,
    y_dm,
    z_dm,
    m_g,
    x_g,
    y_g,
    z_g,
    pot_s=0,
    pot_dm=0,
    pot_g=0,
    eps_dm=0,
    eps_s=0,
    eps_g=0,
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
    x = np.hstack((x_s, x_dm, x_g))
    y = np.hstack((y_s, y_dm, y_g))
    z = np.hstack((z_s, z_dm, z_g))
    m = np.hstack((m_s, m_dm, m_g))
    eps = np.max([eps_dm, eps_s, eps_g])

    total_potential = pot_dm

    if np.all(total_potential == 0.0):
        pot = potential(
            x,
            y,
            z,
            m,
            eps,
        )

        num_s = len(m_s)
        num = len(m_s) + len(m_dm)
        pot_dark = pot[num_s:num]
    else:
        pot_dark = pot_dm

    argmin = pot_dark.argmin()
    x_s = x_s - x_dm[argmin]
    y_s = y_s - y_dm[argmin]
    z_s = z_s - z_dm[argmin]

    x_dm = x_dm - x_dm[argmin]
    y_dm = y_dm - y_dm[argmin]
    z_dm = z_dm - z_dm[argmin]

    x_g = x_g - x_dm[argmin]
    y_g = y_g - y_dm[argmin]
    z_g = z_g - z_dm[argmin]

    return x_s, y_s, z_s, x_dm, y_dm, z_dm, x_g, y_g, z_g
