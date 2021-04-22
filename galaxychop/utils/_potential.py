# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Different potential implementations."""

# =============================================================================
# IMPORTS
# =============================================================================

import astropy.units as u

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

G = (4.299e-6 * u.kpc * (u.km / u.s) ** 2 / u.M_sun).to_value()


# =============================================================================
# BACKENDS
# =============================================================================


def numpy_potential(x, y, z, m, eps):
    """Numpy implementation for the gravitational potential energy calculation.

    Parameters
    ----------
    x, y, z : `np.ndarray`
        Positions of particles. Shape(n,1)
    m : `np.ndarray`
        Masses of particles. Shape(n,1)
    eps : `float`, optional
        Softening parameter. Shape(1,)

    Returns
    -------
    np.ndarray : `float`
    Specific potential energy of particles.

    """
    dist = np.sqrt(
        np.square(x - x.reshape(-1, 1))
        + np.square(y - y.reshape(-1, 1))
        + np.square(z - z.reshape(-1, 1))
        + np.square(eps)
    )

    np.fill_diagonal(dist, 0.0)

    flt = dist != 0
    mdist = np.divide(m, dist, where=flt)

    return mdist.sum(axis=1) * G, np.asarray


# =============================================================================
# API
# =============================================================================

POTENTIAL_BACKENDS = {
    "numpy": numpy_potential,
}


def potential(x, y, z, m, eps=0.0, backend="numpy"):
    """
    Potential energy calculation.

    Given the positions and masses of particles, calculate
    their specific gravitational potential energy.

    Parameters
    ----------
    x, y, z : `np.ndarray`
        Positions of particles. Shape(n,1)
    m : `np.ndarray`
        Masses of particles. Shape(n,1)
    eps : `float`, default value = 0
        Softening parameter. Shape(1,)

    Returns
    -------
    potential : `np.ndarray`
        Specific potential energy of particles. Shape(n,1)
    """
    # extract the implementation
    backend_function = POTENTIAL_BACKENDS[backend]

    # convert all the inputs to float32
    x_f32 = np.asarray(x, dtype=np.float32)
    y_f32 = np.asarray(y, dtype=np.float32)
    z_f32 = np.asarray(z, dtype=np.float32)
    m_f32 = np.asarray(m, dtype=np.float32)
    eps_f32 = np.asarray(eps, dtype=np.float32)

    # execute the function and return
    pot, postproc = backend_function(x_f32, y_f32, z_f32, m_f32, eps_f32)

    # apply the post process and return
    return postproc(pot)
