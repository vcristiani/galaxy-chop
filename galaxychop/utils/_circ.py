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

import warnings

import attr

import numpy as np

import uttr

from ..data import ParticleSetType


# =============================================================================
# CONSTANTS
# =============================================================================


DEFAULT_CBIN = (0.05, 0.005)
"""Default bining of circularity.

Please check the documentation of ``jcirc()``.

"""


# =============================================================================
# API
# =============================================================================
@uttr.s(frozen=True, slots=True)
class JCirc:
    """Circularity information about the stars particles of a galaxy.

    Parameters
    ----------
    normalized_star_energy: np.array
        Normalized specific energy of stars.
    normalized_star_Jz: np.array
        z-component normalized specific angular momentum of the stars.
    eps: np.array
        Circularity parameter (eps : J_z/J_circ).
    eps_r: np.array
        Projected circularity parameter (eps_r: J_p/J_circ).
    x: np.array
        Normalized specific energy for the particle with the maximum
        z-component of the normalized specific angular momentum per bin.
    y: np.array
        Maximum value of the z-component of the normalized specific angular
        momentum per bin.

    """

    normalized_star_energy = uttr.ib()
    normalized_star_Jz = uttr.ib()
    eps = uttr.ib()
    eps_r = uttr.ib()

    x = uttr.ib(metadata={"asdict": False})
    y = uttr.ib(metadata={"asdict": False})

    @classmethod
    def circularity_attributes(cls):
        """Retrieve all the circularity attributes stored in the JCirc class.

        This method returns a tuple of str ignoring those that are marked as
        "asdict=False".

        """
        fields = [
            f.name for f in attr.fields(cls) if f.metadata.get("asdict", True)
        ]
        fields.sort()
        return tuple(fields)

    def as_dict(self):
        """Convert the instance to a dict.

        Attributes are ignored if they are marked as "asdict=False".

        """
        return attr.asdict(
            self, filter=lambda a, v: a.metadata.get("asdict", True)
        )

    def isfinite(self):
        """Return a mask of which elements are finite in all attributes.

        Attributes are ignored if they are marked as "asdict=False".

        """
        selfd = self.as_dict()
        return np.all([np.isfinite(v) for v in selfd.values()], axis=0)


def _jcirc(galaxy, bin0, bin1):
    # this function exists to silence the warnings in the public one

    # extract only the needed columns
    df = galaxy.to_dataframe(
        attributes=["ptypev", "total_energy", "Jx", "Jy", "Jz"]
    )

    Jr_part = np.sqrt(df.Jx.values**2 + df.Jy.values**2)
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
    Jr_star = np.sqrt(df_star.Jx.values**2 + df_star.Jy.values**2)
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

    E_star_norm_ = np.full(len(Etot_s), np.nan)
    Jz_star_norm_ = np.full(len(Jz_star_norm), np.nan)
    eps_ = np.full(len(Etot_s), np.nan)
    eps_r_ = np.full(len(Etot_s), np.nan)

    E_star_norm_[bound_star] = E_star_norm
    Jz_star_norm_[bound_star] = Jz_star_norm
    eps_[bound_star] = eps
    eps_r_[bound_star] = eps_r

    # We remove particles that have circularity < -1 and circularity > 1.
    mask = np.where(eps_ > 1.0)[0]
    E_star_norm_[mask] = np.nan
    Jz_star_norm_[mask] = np.nan
    eps_[mask] = np.nan
    eps_r_[mask] = np.nan

    mask = np.where(eps_ < -1.0)[0]
    E_star_norm_[mask] = np.nan
    Jz_star_norm_[mask] = np.nan
    eps_[mask] = np.nan
    eps_r_[mask] = np.nan

    return JCirc(
        normalized_star_energy=E_star_norm_,
        normalized_star_Jz=Jz_star_norm_,
        eps=eps_,
        eps_r=eps_r_,
        x=x,
        y=y,
    )


def jcirc(
    galaxy,
    bin0=DEFAULT_CBIN[0],
    bin1=DEFAULT_CBIN[1],
    runtime_warnings="ignore",
):
    """
    Calculate galaxy stars particles circularity information.

    Calculation of Normalized specific energy of the stars, z-component
    normalized specific angular momentum of the stars, circularity parameter,
    projected circularity parameter, and the points to build the function of
    the circular angular momentum.

    Parameters
    ----------
    galaxy : ``Galaxy class`` object
    bin0 : float. Default=0.05
        Size of the specific energy bin of the inner part of the galaxy,
        in the range of (-1, -0.1) of the normalized energy.
    bin1 : float. Default=0.005
        Size of the specific energy bin of the outer part of the galaxy,
        in the range of (-0.1, 0) of the normalized energy.
    runtime_warnings : Any warning filter action (default "ignore")
        jcirc usually launches RuntimeWarning during the eps calculation
        because there may be some particle with jcirc=0.
        By default the function decides to ignore these warnings.
        `runtime_warnings` can be set to any valid "action" in the python
        warnings module.

    Return
    ------
    JCirc :
        Circularity attributes of the star components of the galaxy

    Notes
    -----
    The `x` and `y` are calculated from the binning in the normalized specific
    energy. In each bin, the particle with the maximum value of z-component of
    standardized specific angular momentum is selected. This value is assigned
    to the `y` parameter and its corresponding normalized specific energy pair
    value to `x`.

    Examples
    --------
    This returns the normalized specific energy of stars (E_star_norm), the
    z-component normalized specific angular momentum of the stars
    (Jz_star_norm), the circularity parameters (eps : J_z/J_circ and
    eps_r: J_p/J_circ), and the normalized specific energy for the particle
    with the maximum z-component of the normalized specific angular momentum
    per bin (`x`) and the maximum value of the z-component of the normalized
    specific angular momentum per bin (`y`).

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> E_star_norm, Jz_star_norm, eps, eps_r, x, y =
    >>>        galaxy.jcir(bin0=0.05, bin1=0.005)

    """
    with warnings.catch_warnings():
        warnings.simplefilter(runtime_warnings, category=RuntimeWarning)
        return _jcirc(galaxy, bin0, bin1)
