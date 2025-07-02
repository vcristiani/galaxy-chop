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

import numba as nb

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


# =============================================================================
# BACKENDS
# =============================================================================


# GRISPY ======================================================================
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


# 2 Numpy ======================================================================
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


# NUMBA IMPLEMENTATION ========================================================


_numba_eager_signature = nb.float32[:](
    nb.float32[:],
    nb.float32[:],
    nb.float32[:],
    nb.float32[:],
    nb.float32,
)


@nb.jit(_numba_eager_signature, nopython=True, parallel=True, fastmath=False)
def _numba_potential(x, y, z, m, softening):
    n = len(x)
    potential_energy = np.zeros(n, dtype=nb.float32)
    soft2 = nb.float32(softening * softening)

    for i in nb.prange(n):
        pe_i = nb.float32(0.0)  # Forzar float32
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]

        for j in range(n):
            if i != j:
                dx = x_i - x[j]
                dy = y_i - y[j]
                dz = z_i - z[j]

                dist_sq = dx * dx + dy * dy + dz * dz + soft2
                dist = nb.float32(np.sqrt(dist_sq))  # Forzar float32

                pe_i = pe_i + m[j] / dist

        potential_energy[i] = pe_i

    return potential_energy


def numba_potential(x, y, z, m, softening):
    """Wrap the Numba implementation of the gravitational potential.

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
    soft = np.float32(softening)
    epot = _numba_potential(x, y, z, m, soft)

    return epot * const.G, np.asarray


# =============================================================================
# API FUNCTIONS
# =============================================================================


POTENTIAL_BACKENDS = {
    "grispy": grispy_potential,
    "numpy": numpy_potential,
    "numba": numba_potential,
}

DEFAULT_POTENTIAL_BACKEND = "numba"


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


# =============================================================================
# POTENTIALIZER CLASS
# =============================================================================


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
