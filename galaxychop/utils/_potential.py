# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
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

import astropy.constants as c
import astropy.units as u

import numpy as np

from .. import data

try:
    from .fortran import potential as potential_f
except ImportError:
    potential_f = None

POTENTIAL_BACKEND = "numpy" if potential_f is None else "fortran"


# =============================================================================
# CONSTANTS
# =============================================================================

#: GalaxyChop Gravitational unit
G_UNIT = (u.km ** 2 * u.kpc) / (u.s ** 2 * u.solMass)

#: Gravitational constant as float in G_UNIT
G = c.G.to(G_UNIT).to_value()


# =============================================================================
# BACKENDS
# =============================================================================


def fortran_potential(x, y, z, m, softening):
    """Wrap the Fortran implementation of the gravitational potential.

    Parameters
    ----------
    x, y, z : `np.ndarray`
        Positions of particles. Shape(n,1)
    m : `np.ndarray`
        Masses of particles. Shape(n,1)
    softening : `float`, optional
        Softening parameter. Shape(1,)

    Returns
    -------
    np.ndarray : `float`
        Specific potential energy of particles.

    """
    soft = np.asarray(softening)
    epot = potential_f.fortran_potential(x, y, z, m, soft)

    return epot * G, np.asarray


def numpy_potential(x, y, z, m, softening):
    """Numpy implementation for the gravitational potential energy calculation.

    Parameters
    ----------
    x, y, z : `np.ndarray`
        Positions of particles. Shape(n,1)
    m : `np.ndarray`
        Masses of particles. Shape(n,1)
    softening : `float`, optional
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
        + np.square(softening)
    )

    np.fill_diagonal(dist, 0.0)

    flt = dist != 0
    mdist = np.divide(m, dist, where=flt)

    return mdist.sum(axis=1) * G, np.asarray


# =============================================================================
# API
# =============================================================================

POTENTIAL_BACKENDS = {
    "fortran": fortran_potential,
    "numpy": numpy_potential,
}


def potential(galaxy, backend=POTENTIAL_BACKEND):
    """
    Potential energy calculation.

    Given the positions and masses of particles, calculate
    their specific gravitational potential energy.

    Parameters
    ----------
    galaxy : object of Galaxy class.

    Returns
    -------
    galaxy: new object of Galaxy class.
        A new galaxy object the specific potential energy
        of particles calculated.
    """
    if galaxy.has_potential_:
        raise ValueError("galaxy are already calculated")

    # extract the implementation
    backend_function = POTENTIAL_BACKENDS[backend]

    # convert the galaxy in multiple arrays
    df = galaxy.to_dataframe()
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    z = df.z.to_numpy()
    m = df.m.to_numpy()
    softening = df.softening.max()

    # convert all the inputs to float32
    x_f32 = np.asarray(x, dtype=np.float32)
    y_f32 = np.asarray(y, dtype=np.float32)
    z_f32 = np.asarray(z, dtype=np.float32)
    m_f32 = np.asarray(m, dtype=np.float32)
    softening_f32 = np.asarray(softening, dtype=np.float32)

    # execute the function and return
    pot, postproc = backend_function(x_f32, y_f32, z_f32, m_f32, softening_f32)

    # apply the post process to the final potential
    pot = postproc(pot)

    # recreate a new galaxy
    num_s = len(galaxy.stars)
    num = len(galaxy.stars) + len(galaxy.dark_matter)

    pot_s = pot[:num_s]
    pot_dm = pot[num_s:num]
    pot_g = pot[num:]

    new = data.galaxy_as_kwargs(galaxy)

    new.update(
        potential_s=-pot_s * (u.km / u.s) ** 2,
        potential_dm=-pot_dm * (u.km / u.s) ** 2,
        potential_g=-pot_g * (u.km / u.s) ** 2,
    )

    return data.mkgalaxy(**new)
