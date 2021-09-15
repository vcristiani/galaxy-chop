# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Utilities to center a galaxy."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# =============================================================================
# BACKENDS
# =============================================================================


def center(galaxy):
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
    from .. import core

    if not galaxy.has_potential_:
        raise ValueError("galaxy must has the potential energy")

    # sacamos como dataframe lo unico que vamos a operar
    df = galaxy.to_dataframe(columns=["ptype", "x", "y", "z", "potential"])

    # minimo indice de potencial de todo y sacamos cual es la fila
    minpot_idx = df.potential.argmin()
    min_values = df.iloc[minpot_idx]

    # restamos todas las columnas de posiciones por las de minimo valor de
    # potencial y pisamos en df
    columns = ["x", "y", "z", "potential"]
    df[columns] = df[columns] - min_values[columns]

    # espliteamos el dataframe en tipos
    stars = df[df.ptype == core.ParticleSetType.STARS.value]
    dark_matter = df[df.ptype == core.ParticleSetType.DARK_MATTER.value]
    gas = df[df.ptype == core.ParticleSetType.GAS.value]

    # patch
    new = core.galaxy_as_kwargs(galaxy)

    new.update(
        x_s=stars.x,
        y_s=stars.y,
        z_s=stars.z,
        x_dm=dark_matter.x,
        y_dm=dark_matter.y,
        z_dm=dark_matter.z,
        x_g=gas.x,
        y_g=gas.y,
        z_g=gas.z,
    )

    return core.mkgalaxy(**new)


