# This file is part of the
# galxy-chop project (https://github.com/vcristiani/galaxy-chop).
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# #####################################################
# IMPORTS
# #####################################################

import attr
import numpy as np
from galaxychop import utils
from astropy import units as u
# from scipy.interpolate import InterpolatedUnivariateSpline
# from sklearn.mixture import GaussianMixture
# import random

import dask.array as da

# #####################################################
# CONSTANTS
# #####################################################

G = 4.299e-6 * u.kpc * (u.km / u.s) ** 2 / u.M_sun
G = G.to_value()


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
        Star masses. Units M_sun
    eps_s: `np.float()` Default value = 0.
        Softening radius of star particles. Units: kpc.
    x_dm, y_dm, z_dm: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Dark matter positions. Units: kpc
    vx_dm, vy_dm, vz_dm: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Dark matter velocities. Units: km/s
    m_dm: `np.ndarray(n,1)`
        Dark matter masses. Units M_sun
    eps_dm: `np.float()` Default value = 0.
        Softening radius of dark matter particles. Units: kpc.
    x_g, y_g, z_g: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Gas positions. Units: kpc
    vx_g, vy_g, vz_g: `np.ndarray(n,1), np.ndarray(n,1), np.ndarray(n,1)`
        Gas velocities. Units: km/s
    m_g: `np.ndarray(n,1)`
        Gas masses. Units M_sun
    eps_g: `np.float()` Default value = 0.
        Softening radius of gas particles. Units: kpc.
    Etot_s: `np.ndarray(n,1)`
        Total energy of star particles. Units: kg(km/s)**2
    Etot_dm: `np.ndarray(n,1)`
        Total energy of dark matter particles. Units: kg(km/s)**2
    Etot_g: `np.ndarray(n,1)`
        Total energy of gas particles. Units: kg(km/s)**2
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

    x_s = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    y_s = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    z_s = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vx_s = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vy_s = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vz_s = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    m_s = attr.ib(validator=attr.validators.instance_of(u.Quantity))

    x_dm = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    y_dm = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    z_dm = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vx_dm = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vy_dm = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vz_dm = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    m_dm = attr.ib(validator=attr.validators.instance_of(u.Quantity))

    x_g = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    y_g = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    z_g = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vx_g = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vy_g = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    vz_g = attr.ib(validator=attr.validators.instance_of(u.Quantity))
    m_g = attr.ib(validator=attr.validators.instance_of(u.Quantity))

    eps_s = attr.ib(default=0. * u.kpc,
                    validator=attr.validators.instance_of(u.Quantity))
    eps_dm = attr.ib(default=0. * u.kpc,
                     validator=attr.validators.instance_of(u.Quantity))
    eps_g = attr.ib(default=0. * u.kpc,
                    validator=attr.validators.instance_of(u.Quantity))

    Etot_dm = attr.ib(default=None)
    Etot_s = attr.ib(default=None)
    Etot_g = attr.ib(default=None)

    components_s = attr.ib(default=None)
    components_g = attr.ib(default=None)
    metadata = attr.ib(default=None)

    def __change_units_to_array__(f):
        def new_method(self, *args, **kwargs):
            self.x_s = self.x_s.to_value(u.kpc)
            self.y_s = self.y_s.to_value(u.kpc)
            self.z_s = self.z_s.to_value(u.kpc)
            self.vx_s = self.vx_s.to_value(u.km / u.s)
            self.vy_s = self.vy_s.to_value(u.km / u.s)
            self.vz_s = self.vz_s.to_value(u.km / u.s)
            self.m_s = self.m_s.to_value(u.M_sun)
            self.eps_s = self.eps_s.to_value(u.kpc)

            self.x_dm = self.x_dm.to_value(u.kpc)
            self.y_dm = self.y_dm.to_value(u.kpc)
            self.z_dm = self.z_dm.to_value(u.kpc)
            self.vx_dm = self.vx_dm.to_value(u.km / u.s)
            self.vy_dm = self.vy_dm.to_value(u.km / u.s)
            self.vz_dm = self.vz_dm.to_value(u.km / u.s)
            self.m_dm = self.m_dm.to_value(u.M_sun)
            self.eps_dm = self.eps_dm.to_value(u.kpc)

            self.x_g = self.x_g.to_value(u.kpc)
            self.y_g = self.y_g.to_value(u.kpc)
            self.z_g = self.z_g.to_value(u.kpc)
            self.vx_g = self.vx_g.to_value(u.km / u.s)
            self.vy_g = self.vy_g.to_value(u.km / u.s)
            self.vz_g = self.vz_g.to_value(u.km / u.s)
            self.m_g = self.m_g.to_value(u.M_sun)
            self.eps_g = self.eps_g.to_value(u.kpc)

            return f(self, *args, **kwargs)
        return new_method

    @__change_units_to_array__
    def energy(self):

        '''Calculation of kinetic and potencial energy of
        dark matter, star and gas particles'''

        x = np.hstack((self.x_s, self.x_dm, self.x_g))
        y = np.hstack((self.y_s, self.y_dm, self.y_g))
        z = np.hstack((self.z_s, self.z_dm, self.z_g))
        m = np.hstack((self.m_s, self.m_dm, self.m_g))
        eps = np.max(self.eps_dm, self.eps_s, self.eps_s)

        pot = utils.potential(da.asarray(x, chunks=100),
                              da.asarray(y, chunks=100),
                              da.asarray(z, chunks=100),
                              da.asarray(m, chunks=100),
                              da.asarray(eps))

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
