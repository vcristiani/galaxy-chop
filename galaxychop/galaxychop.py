# This file is part of the
#   galxy-chop project (https://github.com/vcristiani/galaxy-chop).
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# #####################################################
# IMPORTS
# #####################################################

import attr
import numpy as np
from galaxychop import utils
# from scipy.interpolate import InterpolatedUnivariateSpline
# from sklearn.mixture import GaussianMixture
# import random

import dask.array as da

# #####################################################
# CONSTANTS
# #####################################################

G = 4.299e+4

# #####################################################
# GALAXY CLASS
# #####################################################


@attr.s(frozen=False)
class Galaxy:
    """This class builds a galaxy object from the masses, positions, and
    velocities of the particles (stars, dark matter, and gas).
    Parameters
    ----------
    x_s, y_s, z_s: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Star positions. Units: kpc
    vx_s, vy_s, vz_s: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Star velocities. Units: km/s
    m_s: `np.ndarray(n,1)`
        Star masses. Units 1e10 M_sun
    x_dm, y_dm, z_dm: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Dark matter positions. Units: kpc
    vx_dm, vy_dm, vz_dm: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Dark matter velocities. Units: km/s
    m_dm: `np.ndarray(n,1)`
        Dark matter masses. Units 1e10 M_sun
    x_g, y_g, z_g: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Gas positions. Units: kpc
    vx_g, vy_g, vz_g: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Gas velocities. Units: km/s
    m_g: `np.ndarray(n,1)`
        Gas masses. Units 1e10 M_sun
    components_s: `np.ndarray(n_star,1)`
        This indicates the component to which the stellar particle is assigned.
        This is chosen as the most probable component.
        `n_star` is the number of stellar particles.
    components_g: `np.ndarray(n_gas,1)`
        This indicates the component to which the gas particle is assigned.
        This is chosen as the most probable component.
        `n_gas` is the number of gas particles.
    Atributes
    ---------
    """

    x_s = attr.ib()
    y_s = attr.ib()
    z_s = attr.ib()
    vx_s = attr.ib()
    vy_s = attr.ib()
    vz_s = attr.ib()
    m_s = attr.ib()

    x_dm = attr.ib()
    y_dm = attr.ib()
    z_dm = attr.ib()
    vx_dm = attr.ib()
    vy_dm = attr.ib()
    vz_dm = attr.ib()
    m_dm = attr.ib()

    x_g = attr.ib()
    y_g = attr.ib()
    z_g = attr.ib()
    vx_g = attr.ib()
    vy_g = attr.ib()
    vz_g = attr.ib()
    m_g = attr.ib()

    E_tot_dark = attr.ib(default=None)
    E_tot_star = attr.ib(default=None)
    E_tot_gas = attr.ib(default=None)

    components_s = attr.ib(default=None)
    components_g = attr.ib(default=None)
    metadata = attr.ib(default=None)


    def energy(self, eps=0):
        '''Calculation of kinetic and potencial energy of
        dark matter, star and gas particles'''

        x = np.hstack((self.x_s, self.x_dm, self.x_g))
        y = np.hstack((self.y_s, self.y_dm, self.y_g))
        z = np.hstack((self.z_s, self.z_dm, self.z_g))
        m = np.hstack((self.m_s, self.m_dm, self.m_g))

        a = utils.potential_dask(da.asarray(x, chunks=100),
                                 da.asarray(y, chunks=100),
                                 da.asarray(z, chunks=100),
                                 da.asarray(m, chunks=100),
                                 da.asarray(eps))

        pot = a.compute()

        pot_star = pot[:len(self.m_s)]
        pot_dark = pot[len(self.m_s):len(self.m_s) + len(self.m_dm)]
        pot_gas = pot[len(self.m_s) + len(self.m_dm):]

        k_dm = 0.5 * (self.vx_dm**2 + self.vy_dm**2 + self.vz_dm**2)
        k_s = 0.5 * (self.vx_s**2 + self.vy_s**2 + self.vz_s**2)
        k_g = 0.5 * (self.vx_g**2 + self.vy_g**2 + self.vz_g**2)

        E_tot_dark = k_dm - pot_dark
        E_tot_star = k_s - pot_star
        E_tot_gas = k_g - pot_gas

        setattr(self, "E_tot_dark", E_tot_dark)
        setattr(self, "E_tot_star", E_tot_star)
        setattr(self, "E_tot_gas", E_tot_gas)

        return E_tot_dark, E_tot_star, E_tot_gas
