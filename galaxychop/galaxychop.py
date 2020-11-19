# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module galaxy-chop."""

# #####################################################
# IMPORTS
# #####################################################

import attr
import numpy as np
from galaxychop import utils
from astropy import units as u
import dask.array as da
import uttr

# from scipy.interpolate import InterpolatedUnivariateSpline
# from sklearn.mixture import GaussianMixture
# import random

# #####################################################
# CONSTANTS
# #####################################################

"""Gravitational constant G.

Units: kpc M_sun^-1 (km/s)^2
"""
G = 4.299e-6


# #####################################################
# GALAXY CLASS
# #####################################################


@attr.s(frozen=False)
class Galaxy:
    """
    Galaxy class.

    Build a galaxy object from the masses, positions, and
    velocities of the particles (stars, dark matter, and gas).

    Parameters
    ----------
    x_s, y_s, z_s: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Star positions. Units: kpc
    vx_s, vy_s, vz_s: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Star velocities. Units: km/s
    m_s: `np.ndarray(n,1)`
        Star masses. Units M_sun
    eps_s: `np.float()` Default value = 0
        Softening radius of star particles. Units: kpc.
    x_dm, y_dm, z_dm: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Dark matter positions. Units: kpc
    vx_dm, vy_dm, vz_dm: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Dark matter velocities. Units: km/s
    m_dm: `np.ndarray(n,1)`
        Dark matter masses. Units M_sun
    eps_dm: `np.float()` Default value = 0
        Softening radius of dark matter particles. Units: kpc
    x_g, y_g, z_g: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Gas positions. Units: kpc
    vx_g, vy_g, vz_g: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Gas velocities. Units: km/s
    m_g: `np.ndarray(n,1)`
        Gas masses. Units M_sun
    eps_g: `np.float()` Default value = 0
        Softening radius of gas particles. Units: kpc
    Etot_s: `np.ndarray(n,1)`
        Total energy of star particles. Units: kg(km/s)**2
    Etot_dm: `np.ndarray(n,1)`
        Total energy of dark matter particles. Units: kg(km/s)**2
    Etot_g: `np.ndarray(n,1)`
        Total energy of gas particles. Units: kg(km/s)**2
    components_s: `np.ndarray(n_star,1)`
        Indicate the component to which the stellar particle is assigned.
        Is chosen as the most probable component.
        `n_star` is the number of stellar particles.
    components_g: `np.ndarray(n_gas,1)`
        Indicate the component to which the gas particle is assigned.
        Is chosen as the most probable component
        `n_gas` is the number of gas particles

    Atributes
    ---------
    """

    x_s = uttr.ib(unit=u.kpc)
    y_s = uttr.ib(unit=u.kpc)
    z_s = uttr.ib(unit=u.kpc)
    vx_s = uttr.ib(unit=(u.km / u.s))
    vy_s = uttr.ib(unit=(u.km / u.s))
    vz_s = uttr.ib(unit=(u.km / u.s))
    m_s = uttr.ib(unit=u.Msun)

    x_dm = uttr.ib(unit=u.kpc)
    y_dm = uttr.ib(unit=u.kpc)
    z_dm = uttr.ib(unit=u.kpc)
    vx_dm = uttr.ib(unit=(u.km / u.s))
    vy_dm = uttr.ib(unit=(u.km / u.s))
    vz_dm = uttr.ib(unit=(u.km / u.s))
    m_dm = uttr.ib(unit=u.Msun)

    x_g = uttr.ib(unit=u.kpc)
    y_g = uttr.ib(unit=u.kpc)
    z_g = uttr.ib(unit=u.kpc)
    vx_g = uttr.ib(unit=(u.km / u.s))
    vy_g = uttr.ib(unit=(u.km / u.s))
    vz_g = uttr.ib(unit=(u.km / u.s))
    m_g = uttr.ib(unit=u.Msun)

    eps_s = uttr.ib(default=0., unit=u.kpc)
    eps_dm = uttr.ib(default=0., unit=u.kpc)
    eps_g = uttr.ib(default=0., unit=u.kpc)

    Etot_dm = uttr.ib(default=0., unit=(u.Msun * (u.km / u.s)**2))
    Etot_s = uttr.ib(default=0., unit=(u.Msun * (u.km / u.s)**2))
    Etot_g = uttr.ib(default=0., unit=(u.Msun * (u.km / u.s)**2))

    to_array = uttr.array_accessor()

    # components_s = attr.ib(default=None)
    # components_g = attr.ib(default=None)
    # metadata = attr.ib(default=None)

    def energy(self):
        """
        Energy calculation.

        Calculate kinetic and potential energy of dark matter,
        star and gas particles.
        """
        x_s = self.to_array.x_s
        y_s = self.to_array.y_s
        z_s = self.to_array.z_s

        x_g = self.to_array.x_g
        y_g = self.to_array.y_g
        z_g = self.to_array.z_g

        x_dm = self.to_array.x_dm
        y_dm = self.to_array.y_dm
        z_dm = self.to_array.z_dm

        m_s = self.to_array.m_s
        m_g = self.to_array.m_g
        m_dm = self.to_array.m_dm

        eps_s = self.to_array.eps_s
        eps_g = self.to_array.eps_g
        eps_dm = self.to_array.eps_dm

        vx_s = self.to_array.x_s
        vy_s = self.to_array.y_s
        vz_s = self.to_array.z_s

        vx_g = self.to_array.x_g
        vy_g = self.to_array.y_g
        vz_g = self.to_array.z_g

        vx_dm = self.to_array.x_dm
        vy_dm = self.to_array.y_dm
        vz_dm = self.to_array.z_dm

        x = np.hstack((x_s, x_dm, x_g))
        y = np.hstack((y_s, y_dm, y_g))
        z = np.hstack((z_s, z_dm, z_g))
        m = np.hstack((m_s, m_dm, m_g))
        eps = np.max([eps_s, eps_dm, eps_g])

        pot = utils.potential(
            da.asarray(x, chunks=100),
            da.asarray(y, chunks=100),
            da.asarray(z, chunks=100),
            da.asarray(m, chunks=100),
            da.asarray(eps),
        )

        pot_s = pot[:len(m_s)]
        pot_dm = pot[len(m_s):len(m_s) + len(m_dm)]
        pot_g = pot[len(m_s) + len(m_dm):]

        k_dm = 0.5 * (vx_dm ** 2 + vy_dm ** 2 + vz_dm ** 2)
        k_s = 0.5 * (vx_s ** 2 + vy_s ** 2 + vz_s ** 2)
        k_g = 0.5 * (vx_g ** 2 + vy_g ** 2 + vz_g ** 2)

        Etot_dm = k_dm - pot_dm
        Etot_s = k_s - pot_s
        Etot_g = k_g - pot_g

        setattr(self, "Etot_dm", Etot_dm * u.Msun * (u.km / u.s) ** 2)
        setattr(self, "Etot_s", Etot_s * u.Msun * (u.km / u.s) ** 2)
        setattr(self, "Etot_g", Etot_g * u.Msun * (u.km / u.s) ** 2)

        return (Etot_dm * u.Msun * (u.km / u.s)**2,
                Etot_s * u.Msun * (u.km / u.s)**2,
                Etot_g * u.Msun * (u.km / u.s)**2)
