# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to make a grid and run Nearest Neighbor Searchs via GriSPy."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import grispy as gsp

from .. import data

# =============================================================================
# BACKENDS
# =============================================================================


def make_grid(df,n_cells):
    """Space grid making (?).

    Make a grid of the volume and index galaxy's particles to each cell.

    Parameters
    ---------- *Cambiar
    df : ``Galaxy class`` object as dataframe (!)

    Returns
    ------- *Cambiar
    galaxy : new ``Galaxy class`` object
        A new galaxy object with centered positions respect to the position of
        the lowest potential particle.
    """
    #En este caso no hace falta el potencial ¿Pero qué otra cosa si?
    # ¿La glx en sí?
    #if not galaxy.has_potential_:
    #    raise ValueError("galaxy must has the potential energy")

    # Le debería pasar directamente lo que quiero usar, no la glx entera...
    # We extract only the needed column to make the grid
    #df = galaxy.to_dataframe(attributes=["x", "y", "z","m"])

    #Lo dejo para guiarme --
    # minimum potential index of all particles and we extract data frame row
    #minpot_idx = df.potential.argmin()
    #min_values = df.iloc[minpot_idx]
    #-----------------------

    #Como lo tengo escrito yo:
    # Size of the box that contains all particles
    L_box = max(np.abs([max(df['x'])-min(df['x']),
                       max(df['y'])-min(df['y']),
                       max(df['z'])-min(df['z'])]))

    positions = np.array([df['x'],df['y'],df['z']])
    particle_mass = np.array(df['m'])
    positions = np.transpose(positions) # Para que le guste a GriSPy

    # Make the grid (n_cells ~ 2**5 works well for 1e+4 ~ 1e+5 particles)
    grid = gsp.GriSPy(positions, n_cells)

    return positions,particle_mass,L_box,grid

def potential_grispy(centre, positions, particle_mass, 
                     bubble_size, shell_width, L_box, grid):
    """Compute the potential of a particle given the grid and the system.

    Given the particle to compute its potential energy, iteratively
    make shells to aproximate their monopole contribution.
    PD: Pero calro, todas las otras funciones vienen comiéndose "galaxias",
    ¿Debería mantenerme con eso acá? Estas funciones van de la mano, así que ojo
    con mi implementación naïve.

    Parameters
    ----------
    centre : np.array
        3D spatial position of the particle to compute its potential.
    positions : np.array
        3D spatial position of all the galaxy particles.
    particle_mass : np.array
        Individual masses of all the galaxy particles.
    bubble_size : float
        Radii of the sphere that will contain the closest particles,
        which potential contribution will be calculated via direct-
        sumation.
    shell_width : float
        Width of the consecutive shells that will contain further particles.
        The shell's potential contribution will be aproximated by its
        monopole term.
    L_box : float
        Size of the box that contains all particles. This defines the upper
        limit value of the shells to implement the GriSPy's NNS.
    grid : ``GriSPy`` object
        Spatial grid populated by the galaxy particles, needed to do the NNS.

    Returns
    -------
    pot_shells : float
        Potential of the given particle through the shells' monopole aproximation.
    """
    #En este caso no hace falta el potencial ¿Pero qué otra cosa si?
    # ¿La glx en sí?
    #if not galaxy.has_potential_:
    #    raise ValueError("galaxy must has the potential energy")

    #Como lo tengo escrito yo:
    
    # Use the bubble method to find the closest particles
    bubble_dist, bubble_ind = grid.bubble_neighbors(
        centre, distance_upper_bound=bubble_size
    )
    
    # Compute the potential contribution via direct-sumation
    # of these bubble's particles
    pot_shells = 0. # The potential variable.
    for idx,distance in enumerate(bubble_dist[0]):
        if distance > 0.:
            pot_shells -= G * particle_mass[bubble_ind[0][idx]]/distance
        else:
            continue
            
    d_min_shell = bubble_size # Shell's lower limit to initialize the loop
    while d_min_shell < L_box:
        # Use the shell method to populate it
        shell_dist, shell_ind = grid.shell_neighbors(
            centre,
            distance_lower_bound=d_min_shell,
            distance_upper_bound=d_min_shell + shell_width
        )

        # Compute the monopole potential contribution of this shell
        for idx,distance in enumerate(shell_dist[0]):
            # Due to non-periodicity, distance > 0 always
            pot_shells -= G * particle_mass[shell_ind[0][idx]]/distance

        d_min_shell += shell_width # Repeat for next shell (further)
        
    return pot_shells