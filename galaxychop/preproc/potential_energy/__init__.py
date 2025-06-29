# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Module for calculus of potential energy."""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import astropy.units as u

import numpy as np

from .grispy_calculation import (
    make_grid,
    potential_grispy,
)
from .._base import GalaxyTransformerABC
from ... import (
    constants as const,
    core,
)
from ...utils import doc_inherit

try:
    from .fortran import potential as potential_f
except ImportError:
    potential_f = None

#: The default potential backend to use.
DEFAULT_POTENTIAL_BACKEND = "numpy" if potential_f is None else "fortran"


# =============================================================================
# BACKENDS
# =============================================================================


def fortran_potential(x, y, z, m, softening):
    """
    Wrap the Fortran implementation of the gravitational potential.

    Parameters
    ----------
    x, y, z : np.ndarray
        Positions of particles. Shape: (n,1).
    m : np.ndarray
        Masses of particles. Shape: (n,1).
    softening : float, optional
        Softening parameter. Shape: (1,).

    Returns
    -------
    np.ndarray : float
        Specific potential energy of particles.

    """
    soft = np.asarray(softening)
    epot = potential_f.fortran_potential(x, y, z, m, soft)

    return epot * const.G, np.asarray


def grispy_potential(x, y, z, m, softening):
    """
    Grispy implementation of the gravitational potential energy calculation.

    Parameters
    ----------
    x, y, z : np.ndarray
        Positions of particles. Shape: (n,1).
    m : np.ndarray
        Masses of particles. Shape: (n,1).
    softening : float, optional
        Softening parameter. Shape: (1,).

    Returns
    -------
    np.ndarray : float
        Specific potential energy of particles.

    """
    # Make the grid of the system
    l_box, grid = make_grid(x, y, z)

    # For each particle, compute its potential energy
    epot = np.empty(len(m))
    for idx, particle in enumerate(m):
        centre = np.array([[x[idx], y[idx], z[idx]]])
        epot[idx] = potential_grispy(
            centre,
            m,
            bubble_size=5 * softening,
            shell_width=0.1 * l_box,
            l_box=l_box,
            grid=grid,
        )

    return epot * const.G, np.asarray


def numpy_potential(x, y, z, m, softening):
    """
    Numpy implementation for the gravitational potential energy calculation.

    Parameters
    ----------
    x, y, z : np.ndarray
        Positions of particles. Shape: (n,1).
    m : np.ndarray
        Masses of particles. Shape:(n,1).
    softening : float, optional
        Softening parameter. Shape: (1,).

    Returns
    -------
    np.ndarray : float
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
    mdist = np.divide(m, dist, out=np.zeros_like(dist), where=flt)

    return mdist.sum(axis=1) * const.G, np.asarray


# =============================================================================
# POTENTIALIZER CLASS
# =============================================================================

POTENTIAL_BACKENDS = {
    "fortran": fortran_potential,
    "grispy": grispy_potential,
    "numpy": numpy_potential,
}


class Potentializer(GalaxyTransformerABC):
    """
    Potentializer class.

    Given the positions and masses of particles, calculate
    their specific gravitational potential energy.

    Parameters
    ----------
    galaxy : ``Galaxy class`` object
        The galaxy object without the potential energy of particles
    backends : str, default="numpy"
        Method to calculate the potential energy of each particle

    Returns
    -------
    galaxy: new ``Galaxy class`` object
        A new galaxy object with the specific potential energy of particles
        calculated.

    """

    def __init__(self, backend=DEFAULT_POTENTIAL_BACKEND):
        self.backend = backend

        if self.backend not in POTENTIAL_BACKENDS:
            raise TypeError(
                "The backend entered is not in the possible Backends"
            )
        else:
            print("CREATED POTENCIALIZER WITH BACKEND  " + self.backend)
            pass

    @doc_inherit(GalaxyTransformerABC.transform)
    def transform(self, galaxy):
        return potential(galaxy, backend=self.backend)

    @doc_inherit(GalaxyTransformerABC.checker)
    def checker(self, galaxy, **kwargs):
        return galaxy.has_potential_


# =============================================================================
# API FUNCTIONS
# =============================================================================


def potential(galaxy, *, backend=DEFAULT_POTENTIAL_BACKEND):
    """
    Potential energy calculation.

    Given the positions and masses of particles, calculate
    their specific gravitational potential energy as a function.

    Parameters
    ----------
    galaxy : ``Galaxy class`` object

    Returns
    -------
    galaxy: new ``Galaxy class`` object
        A new galaxy object with the specific potential energy of particles
        calculated.
    """
    if galaxy.has_potential_:
        warnings.warn(
            "Galaxy potential is already calculated. \
            Resuming...",
            UserWarning,
        )

    # extract the implementation
    backend_function = POTENTIAL_BACKENDS[backend]

    # convert the galaxy in multiple arrays
    df = galaxy.to_dataframe(attributes=["x", "y", "z", "m", "softening"])
    x = df.x.to_numpy(dtype=np.float32)
    y = df.y.to_numpy(dtype=np.float32)
    z = df.z.to_numpy(dtype=np.float32)
    m = df.m.to_numpy(dtype=np.float32)
    softening = np.asarray(df.softening.max(), dtype=np.float32)

    # cleanup df
    del df

    # execute the function and return
    pot, postproc = backend_function(x, y, z, m, softening)

    # cleanup again
    del x, y, z, m, softening

    # apply the post process to the final potential
    pot = postproc(pot)

    # recreate a new galaxy
    num_s = len(galaxy.stars)
    num = len(galaxy.stars) + len(galaxy.dark_matter)

    pot_s = pot[:num_s]
    pot_dm = pot[num_s:num]
    pot_g = pot[num:]

    new = galaxy.disassemble()

    new.update(
        potential_s=-pot_s * (u.km / u.s) ** 2,
        potential_dm=-pot_dm * (u.km / u.s) ** 2,
        potential_g=-pot_g * (u.km / u.s) ** 2,
    )

    return core.mkgalaxy(**new)
