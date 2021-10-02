# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for Processing energy and angular momentum of the galaxies."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import namedtuple

import numpy as np

from ..data import ParticleSetType


# =============================================================================
# API
# =============================================================================

JCirc = namedtuple("JCirc", ["E_star_norm", "eps", "eps_r", "x", "y"])


def jcirc(galaxy, bin0=0.05, bin1=0.005):

    """
    Processing energy and angular momentum.

    Calculation of Normalized specific energy of the stars,
    circularity parameter calculation, projected circularity parameter,
    and the points to build the function of the circular angular momentum.

    Parameters
    ----------
    galaxy : object of Galaxy class.
    bin0 : `float`. Default=0.05
        Size of the specific energy bin of the inner part of the galaxy,
        in the range of (-1, -0.1) of the normalized energy.
    bin1 : `float`. Default=0.005
        Size of the specific energy bin of the outer part of the galaxy,
        in the range of (-0.1, 0) of the normalized energy.

    Return
    ------
    tuple : `float`
        (E_star_norm, eps, eps_r, x, y): Normalized specific energy of
        the stars, circularity parameter (J_z/J_circ), projected
        circularity parameter (J_p/J_circ), the normalized specific
        energy for the particle with the maximum z-specific angular
        momentum component per the bin (x), and the maximum of z-specific
        angular momentum component (y).
        See section Notes for more details.
        Shape(n_s, 1). Unit: dimensionless

    Notes
    -----
    The `x` and `y` are calculated from the binning in the
    normalized specific energy. In each bin, the particle with the
    maximum value of z-component of standardized specific angular
    momentum is selected. This value is assigned to the `y` parameter
    and its corresponding normalized specific energy pair value to `x`.

    Examples
    --------
    This returns the normalized specific energy of stars (E_star_norm), the
    circularity parameters (eps : J_z/J_circ and
    eps_r: J_p/J_circ), and the normalized specific energy for the particle
    with the maximum z-component of the normalized specific angular
    momentum per bin (`x`) and the maximum value of the z-component of the
    normalized specific angular momentum per bin (`y`).

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> E_star_norm, eps, eps_r, x, y = galaxy.jcir(bin0=0.05, bin1=0.005)
    """

    # extract only the needed columns
    df = galaxy.to_dataframe(
        attributes=["ptypev", "total_energy", "Jx", "Jy", "Jz"]
    )

    Jr_part = np.sqrt(df.Jx.values ** 2 + df.Jy.values ** 2)
    E_tot = df.total_energy.values

    # Remove the particles that are not bound: E > 0 and with E = -inf.
    (bound,) = np.where((E_tot <= 0.0) & (E_tot != -np.inf))

    # Normalize the two variables: E between 0 and 1; Jz between -1 and 1.
    E = E_tot[bound] / np.abs(np.min(E_tot[bound]))
    Jz = df.Jz.values[bound] / np.max(np.abs(df.Jz.values[bound]))

    # Build the specific energy binning and select the Jz values to
    # calculate J_circ.
    aux0 = np.arange(-1.0, -0.1, bin0)
    aux1 = np.arange(-0.1, 0.0, bin1)
    aux = np.concatenate([aux0, aux1], axis=0)

    x = np.zeros(len(aux) + 1)
    y = np.zeros(len(aux) + 1)

    x[0] = -1.0
    y[0] = np.abs(Jz[np.argmin(E)])

    for i in range(1, len(aux)):
        (mask,) = np.where((E <= aux[i]) & (E > aux[i - 1]))
        s = np.argsort(np.abs(Jz[mask]))

        # We take into account whether or not there are particles in the
        # specific energy bins.
        if len(s) != 0:
            if len(s) == 1:
                x[i] = E[mask][s]
                y[i] = np.abs(Jz[mask][s])
            else:
                if (
                    1.0 - (np.abs(Jz[mask][s][-2]) / np.abs(Jz[mask][s][-1]))
                ) >= 0.01:
                    x[i] = E[mask][s][-2]
                    y[i] = np.abs(Jz[mask][s][-2])
                else:
                    x[i] = E[mask][s][-1]
                    y[i] = np.abs(Jz[mask][s][-1])
        else:
            pass

    # Mask to complete the last bin, in case there are no empty bins.
    (mask,) = np.where(E > aux[len(aux) - 1])

    if len(mask):
        x[len(aux)] = E[mask][np.abs(Jz[mask]).argmax()]
        y[len(aux)] = np.abs(Jz[mask][np.abs(Jz[mask]).argmax()])

    # In case there are empty bins, we get rid of them.
    else:
        i = len(np.where(y == 0)[0]) - 1
        if i == 0:
            x = x[:-1]
            y = y[:-1]
        else:
            x = x[:-i]
            y = y[:-i]

    # In case some intermediate bin does not have points.
    (zero,) = np.where(x != 0.0)
    x = x[zero]
    y = y[zero]

    # Stars particles
    df_star = df[df.ptypev == ParticleSetType.STARS.value]
    Jr_star = np.sqrt(df_star.Jx.values ** 2 + df_star.Jy.values ** 2)
    Etot_s = df_star.total_energy.values

    # Remove the star particles that are not bound:
    # E > 0 and with E = -inf.
    (bound_star,) = np.where((Etot_s <= 0.0) & (Etot_s != -np.inf))

    # Normalize E, Jz and Jr for the stars.
    E_star_norm = Etot_s[bound_star] / np.abs(np.min(E_tot[bound]))
    Jz_star_norm = df_star.Jz.values[bound_star] / np.max(
        np.abs(df.Jz.values[bound])
    )
    Jr_star_norm = Jr_star[bound_star] / np.max(np.abs(Jr_part[bound]))

    # Calculates of the circularity parameters Jz/Jcirc and Jproy/Jcirc.
    j_circ = np.interp(E_star_norm, x, y)
    eps = Jz_star_norm / j_circ
    eps_r = Jr_star_norm / j_circ

    # We remove particles that have circularity < -1 and circularity > 1.
    (mask,) = np.where((eps <= 1.0) & (eps >= -1.0))

    E_star_norm_ = np.full(len(Etot_s), np.nan)
    eps_ = np.full(len(Etot_s), np.nan)
    eps_r_ = np.full(len(Etot_s), np.nan)

    E_star_norm_[bound_star[mask]] = E_star_norm[mask]
    eps_[bound_star[mask]] = eps[mask]
    eps_r_[bound_star[mask]] = eps_r[mask]

    return JCirc(E_star_norm=E_star_norm, eps=eps, eps_r=eps_r, x=x, y=y)
