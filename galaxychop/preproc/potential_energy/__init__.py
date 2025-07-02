# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""
Gravitational potential energy calculation module.

This module provides multiple backends for computing the gravitational
potential energy of particles in galaxy simulations, including optimized
implementations using NumPy, Numba, and GriSPy.
"""

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
    Calculate gravitational potential energy using GriSPy algorithm.

    This implementation uses the GriSPy library for efficient neighbor
    searching and potential calculation. GriSPy is particularly efficient
    for large N-body systems with spatially clustered particles.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        X-coordinates of particles in simulation units.
    y : np.ndarray, shape (n,)
        Y-coordinates of particles in simulation units.
    z : np.ndarray, shape (n,)
        Z-coordinates of particles in simulation units.
    m : np.ndarray, shape (n,)
        Masses of particles in simulation units.
    softening : float
        Gravitational softening parameter to avoid singularities at
        small separations. Must be positive.

    Returns
    -------
    epot : np.ndarray, shape (n,)
        Specific gravitational potential energy of each particle.
        Units: [G * M / L] where G is gravitational constant.
    postproc : callable
        Post-processing function (np.asarray) for result formatting.

    Notes
    -----
    The GriSPy algorithm constructs a spatial grid to optimize neighbor
    searches. The bubble size is set to 5 times the softening parameter,
    and shell width is 10% of the box size for optimal performance.

    References
    ----------
    .. [1] Chalela, M. et al. "GriSPy: A Python package for fixed-radius
           nearest neighbors search." Astronomy and Computing, 2021.
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


# NUMPY =======================================================================
def numpy_potential(x, y, z, m, softening):
    """
    Calculate gravitational potential energy using pure NumPy.

    This is a reference implementation using vectorized NumPy operations.
    While straightforward, it has O(N²) memory complexity and may be slow
    for large particle numbers due to the full distance matrix calculation.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        X-coordinates of particles in simulation units.
    y : np.ndarray, shape (n,)
        Y-coordinates of particles in simulation units.
    z : np.ndarray, shape (n,)
        Z-coordinates of particles in simulation units.
    m : np.ndarray, shape (n,)
        Masses of particles in simulation units.
    softening : float
        Gravitational softening parameter to avoid singularities.
        Added to distance calculation as sqrt(r² + ε²).

    Returns
    -------
    epot : np.ndarray, shape (n,)
        Specific gravitational potential energy of each particle.
        Units: [G * M / L] where G is gravitational constant.
    postproc : callable
        Post-processing function (np.asarray) for result formatting.

    Notes
    -----
    The potential energy is calculated as:
    φᵢ = -G * Σⱼ≠ᵢ (mⱼ / √((rᵢ - rⱼ)² + ε²))

    Memory usage scales as O(N²) due to the full distance matrix.
    For large N (>10⁴), consider using the Numba or GriSPy backends.

    Examples
    --------
    >>> x = np.array([0., 1., 2.])
    >>> y = np.array([0., 0., 0.])
    >>> z = np.array([0., 0., 0.])
    >>> m = np.array([1., 1., 1.])
    >>> epot, _ = numpy_potential(x, y, z, m, softening=0.1)
    """
    # Calculate pairwise distances with softening
    dist = np.sqrt(
        np.square(x - x.reshape(-1, 1))
        + np.square(y - y.reshape(-1, 1))
        + np.square(z - z.reshape(-1, 1))
        + np.square(softening)
    )

    # Set diagonal to zero (self-interaction)
    np.fill_diagonal(dist, 0.0)

    # Avoid division by zero and calculate m/r terms
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
    """
    Numba-compiled potential energy calculation kernel.

    This is the low-level computational kernel optimized with Numba JIT
    compilation. It uses parallel execution and explicit float32 precision
    for optimal performance.

    Parameters
    ----------
    x, y, z : np.ndarray, dtype=float32
        Particle coordinates.
    m : np.ndarray, dtype=float32
        Particle masses.
    softening : float32
        Softening parameter.

    Returns
    -------
    potential_energy : np.ndarray, dtype=float32
        Potential energy array.

    Notes
    -----
    This function is compiled with Numba's @jit decorator for performance.
    The parallel=True flag enables automatic parallelization of the outer
    loop using multiple CPU cores.
    """
    n = len(x)
    potential_energy = np.zeros(n, dtype=nb.float32)
    soft2 = nb.float32(softening * softening)

    for i in nb.prange(n):
        pe_i = nb.float32(0.0)
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]

        for j in range(n):
            if i != j:
                dx = x_i - x[j]
                dy = y_i - y[j]
                dz = z_i - z[j]

                dist_sq = dx * dx + dy * dy + dz * dz + soft2
                dist = nb.float32(np.sqrt(dist_sq))

                pe_i = pe_i + m[j] / dist

        potential_energy[i] = pe_i

    return potential_energy


def numba_potential(x, y, z, m, softening):
    """
    Calculate gravitational potential energy using Numba JIT compilation.

    This implementation provides the best performance for CPU-based
    calculations through just-in-time compilation and automatic
    parallelization. It has O(N²) computational complexity but with
    significantly better performance than pure NumPy.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        X-coordinates of particles in simulation units.
    y : np.ndarray, shape (n,)
        Y-coordinates of particles in simulation units.
    z : np.ndarray, shape (n,)
        Z-coordinates of particles in simulation units.
    m : np.ndarray, shape (n,)
        Masses of particles in simulation units.
    softening : float
        Gravitational softening parameter. Prevents singularities when
        particles are very close together.

    Returns
    -------
    epot : np.ndarray, shape (n,)
        Specific gravitational potential energy of each particle.
        Units: [G * M / L] where G is gravitational constant.
    postproc : callable
        Post-processing function (np.asarray) for result formatting.

    Notes
    -----
    This function automatically converts inputs to float32 for optimal
    performance.

    The parallel execution uses all available CPU cores by default.
    For very small systems (N < 1000), the overhead might make this
    slower than the NumPy implementation.


    Examples
    --------
    >>> x = np.random.rand(1000)
    >>> y = np.random.rand(1000)
    >>> z = np.random.rand(1000)
    >>> m = np.ones(1000)
    >>> epot, _ = numba_potential(x, y, z, m, softening=0.01)
    """
    soft = np.float32(softening)
    epot = _numba_potential(x, y, z, m, soft)

    return epot * const.G, np.asarray


# =============================================================================
# API FUNCTIONS
# =============================================================================

# Available backends for potential calculation
POTENTIAL_BACKENDS = {
    "grispy": grispy_potential,
    "numpy": numpy_potential,
    "numba": numba_potential,
}


#: Default backend for potential energy calculations
DEFAULT_POTENTIAL_BACKEND = "numba"


def potential(galaxy, *, backend=DEFAULT_POTENTIAL_BACKEND):
    """
    Calculate gravitational potential energy for all particles in a galaxy.

    This function computes the specific gravitational potential energy for
    each particle (stars, dark matter, gas) in the galaxy using the
    specified computational backend.

    Parameters
    ----------
    galaxy : Galaxy
        Galaxy object containing particle data. Must have position (x, y, z)
        and mass (m) attributes for all particle types.
    backend : {'numba', 'numpy', 'grispy'}, optional
        Computational backend to use. Default is 'numba'.

        - 'numba': JIT-compiled, fastest for most cases
        - 'numpy': Pure NumPy, reference implementation
        - 'grispy': Tree-based algorithm, efficient for clustered data

    Returns
    -------
    new_galaxy : Galaxy
        New Galaxy object with potential energy attributes added:

        - `potential_s` : Potential energy of stellar particles
        - `potential_dm` : Potential energy of dark matter particles
        - `potential_g` : Potential energy of gas particles

        Units are (km/s)² following astropy unit conventions.

    Warns
    -----
    UserWarning
        If potential energy is already calculated for the galaxy.

    Notes
    -----
    The gravitational potential energy is calculated as:

    φᵢ = -G * Σⱼ≠ᵢ (mⱼ / |rᵢ - rⱼ + ε|)

    where G is the gravitational constant, mⱼ are particle masses,
    rᵢ, rⱼ are particle positions, and ε is the softening parameter.

    The softening parameter prevents numerical singularities when particles
    are very close together. It is taken as the maximum softening value
    present in the galaxy data.


    Examples
    --------
    Calculate potential energy using default (numba) backend:

    >>> galaxy_with_potential = potential(galaxy)

    Use GriSPy backend for large, clustered systems:

    >>> galaxy_with_potential = potential(galaxy, backend='grispy')

    Access potential energy of stellar particles:

    >>> stellar_potential = galaxy_with_potential.potential_s

    See Also
    --------
    Potentializer : Class-based interface for potential energy calculation
    """
    if galaxy.has_potential_:
        warnings.warn(
            "Galaxy potential is already calculated. Resuming...",
            UserWarning,
        )

    # Extract the implementation
    backend_function = POTENTIAL_BACKENDS[backend]

    # Convert the galaxy to multiple arrays
    df = galaxy.to_dataframe(attributes=["x", "y", "z", "m", "softening"])
    x = df.x.to_numpy(dtype=np.float32)
    y = df.y.to_numpy(dtype=np.float32)
    z = df.z.to_numpy(dtype=np.float32)
    m = df.m.to_numpy(dtype=np.float32)
    softening = np.asarray(df.softening.max(), dtype=np.float32)

    # Cleanup df
    del df

    # Execute the function and return
    pot, postproc = backend_function(x, y, z, m, softening)

    # Cleanup again
    del x, y, z, m, softening

    # Apply the post process to the final potential
    pot = postproc(pot)

    # Recreate a new galaxy
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
    Galaxy transformer for gravitational potential energy calculation.

    This class provides a scikit-learn style interface for computing
    gravitational potential energy of galaxy particles. It implements
    the transformer pattern with fit/transform methods.

    Parameters
    ----------
    backend : {'numba', 'numpy', 'grispy'}, optional
        Computational backend to use for potential calculation.
        Default is 'numba'.

        Available backends:

        - 'numba': JIT-compiled implementation, fastest for most cases
        - 'numpy': Pure NumPy implementation, reference but with higher
                   memory usage.
        - 'grispy': Tree-based algorithm, memory efficient for large N

    Attributes
    ----------
    backend : str
        The computational backend being used.

    Raises
    ------
    TypeError
        If the specified backend is not available.

    Notes
    -----
    This class follows the transformer pattern common in machine learning
    libraries. The `transform` method performs the actual computation,
    while `checker` verifies if the transformation has already been applied.

    The class is particularly useful for building analysis pipelines where
    potential energy calculation is one step among many transformations.


    """

    def __init__(self, backend=DEFAULT_POTENTIAL_BACKEND):
        self.backend = backend

        if self.backend not in POTENTIAL_BACKENDS:
            available_backends = list(POTENTIAL_BACKENDS.keys())
            raise TypeError(
                f"Backend '{self.backend}' is not available. "
                f"Available backends: {available_backends}"
            )

        print(f"Created Potentializer with backend: {self.backend}")

    @doc_inherit(GalaxyTransformerABC.transform)
    def transform(self, galaxy):
        """
        Transform galaxy by calculating potential energy.

        Parameters
        ----------
        galaxy : Galaxy
            Input galaxy object to transform.

        Returns
        -------
        Galaxy
            New galaxy object with potential energy calculated.
        """
        return potential(galaxy, backend=self.backend)

    @doc_inherit(GalaxyTransformerABC.checker)
    def checker(self, galaxy, **kwargs):
        """
        Check if potential energy has already been calculated.

        Parameters
        ----------
        galaxy : Galaxy
            Galaxy object to check.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        bool
            True if potential energy is already calculated, False otherwise.
        """
        return galaxy.has_potential_
