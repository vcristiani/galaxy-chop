# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

#Bruno: Acá meto mano (importar GriSPy y def los potenciales nuevos)

"""Different potential implementations."""

# =============================================================================
# IMPORTS
# =============================================================================

import astropy.constants as c
import astropy.units as u

import numpy as np

# New. Necesito GriSPy --
import grispy as gsp
# -----------------------

from .. import data

# New. ¿Debería importar las funciones básicas para
# realizar el grid + cálculo o escribo todo acá?
# Supongo opción 1, así que creo un .py con las ~3 func
# fundamentales... ¿Tocar __init__.py? ---------------
from gridsearch import make_grid, potential_grispy
# ¿Así? ----------------------------------------------

try:
    from .fortran import potential as potential_f
except ImportError:
    potential_f = None

# New. Sin existir, pero así debería ser (?): --
try:
    from .C import potencial as potential_c
except ImportError:
    potential_c = None
#-----------------------------------------------  

#: The default potential backend to use.
DEFAULT_POTENTIAL_BACKEND = "numpy" if potential_f is None or \
    potential_c is None else "fortran"


# =============================================================================
# CONSTANTS
# =============================================================================

#: GalaxyChop Gravitational unit
G_UNIT = (u.km**2 * u.kpc) / (u.s**2 * u.solMass)

#: Gravitational constant as float in G_UNIT
G = c.G.to(G_UNIT).to_value()


# =============================================================================
# BACKENDS
# =============================================================================


def fortran_potential(x, y, z, m, softening):
    """Wrap the Fortran implementation of the gravitational potential.

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

    return epot * G, np.asarray


def numpy_potential(x, y, z, m, softening):
    """Numpy implementation for the gravitational potential energy calculation.

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

    return mdist.sum(axis=1) * G, np.asarray


def grispy_potential(galaxy,n_cells,bubble_size,shell_width):
    """GriSPy implementation for the gravitational potential energy calculation
    from the bubble and shell NNS methods of this library.

    galaxy : ``Galaxy class`` object

    Returns
    -------
    galaxy: new ``Galaxy class`` object
        A new galaxy object with the specific potential energy of particles
        calculated.

    """
    # convert the galaxy in multiple arrays
    df = galaxy.to_dataframe(attributes=["x", "y", "z", "m"])

    n_particles = len(df)

    # Make the grid of the space and populate its cells
    positions,particle_mass,L_box,grid = make_grid(df,n_cells)

    # Initialize the NumPy array for the potential of every particle
    epot = np.empty(n_particles)

    # Calculate the potential through shell aproximation of every particle
    for idx,part in enumerate(positions):
        # The centre (particle) for this step
        centre = np.array([part])
        
        # Compute the potential (in [(km/s)^2])
        pot_shells = potential_grispy(
            centre,positions,particle_mass,
            bubble_size,shell_width,L_box,grid)
        
        # Assign it to the particle
        epot[idx] = pot_shells
        
        # A % counter if necessary
        #if idx % 100 == 0:
        #    print(f'{100* idx/n_particles:.2f} %')

    return epot, np.asarray


def octree_potential(x, y, z, m, softening):
    """Wrap the C implementation of the gravitational potential calculation
    using an Octree algorithm.

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

    # Según lo que me pasó Luisito:
    #calcula_potencial(Npoints,np.full(Npoints,1.4e+6,dtype=np.float32),
    #                       np.float32(data[:,0]),np.float32(data[:,1]),np.float32(data[:,2]))
    # Lo pongo a lo naïve y siguiendo los lineamientos de caja negra que es:
    # (revisar el utils.py y el potencial.c de la carpeta creada "C").
    epot = potential_c.calcula_potential(len(x),m,x, y, z) # Ya multiplicado por G y en
                                                           # unidades [(km/s)^2]
    #Ojo con esto último, porque en la función siguiente hace algo con las unidades...                                                     

    return epot, np.asarray


# =============================================================================
# API
# =============================================================================

POTENTIAL_BACKENDS = {
    "fortran": fortran_potential,
    "numpy": numpy_potential,
    "octree": octree_potential,
}


#: The default potential backend to use.
def potential(galaxy, backend=DEFAULT_POTENTIAL_BACKEND):
    """
    Potential energy calculation.

    Given the positions and masses of particles, calculate
    their specific gravitational potential energy.

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
        raise ValueError("galaxy potential are already calculated")

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

    new = data.galaxy_as_kwargs(galaxy)

    # Esto. Ojo porque el Octree ya devuelve [(km/s)^2]
    new.update(
        potential_s=-pot_s * (u.km / u.s) ** 2,
        potential_dm=-pot_dm * (u.km / u.s) ** 2,
        potential_g=-pot_g * (u.km / u.s) ** 2,
    )

    return data.mkgalaxy(**new)
