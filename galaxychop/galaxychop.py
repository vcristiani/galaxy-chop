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
from utils import aling, rot
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.mixture import GaussianMixture
import random

# #####################################################
# GALAXY CLASS
# #####################################################


@attr.s(frozen=True)
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

    components_s = attr.ib(default=None)
    components_g = attr.ib(default=None)
    metadata = attr.ib(default=None)

    def energy(self, pot_star, pot_dark, pot_gas):
        '''Calculation of kinetic and potencial energy of
        dark matter, star and gas particles'''

        k_dm = 0.5 * (self.vx_dm**2 + self.vy_dm**2 + self.vz_dm**2)
        k_s = 0.5 * (self.vx_s**2 + self.vy_s**2 + self.vz_s**2)
        k_g = 0.5 * (self.vx_g**2 + self.vy_g**2 + self.vz_g**2)

        E_tot_dark = k_dm - self.pot_dark[:, 1]
        E_tot_star = k_s - self.pot_star[:, 1]
        E_tot_gas = k_g - self.pot_gas[:, 1]

        return E_tot_star, E_tot_dark, E_tot_gas

    def angular_momentum(self):
        """Components of the angular momentum of stars, gas and DM.

        Parameters
        ----------

        Coordinates and velocities for stars: x_s, y_s, z_s and
         vx_s, vy_s, vz_s.

        Coordinates and velocities for dark matter: x_dm, y_dm, z_dm and
         vx_dm, vy_dm, vz_dm.

        Coordinates and velocities for gas: x_g, y_g, z_g and vx_g, vy_g, vz_g.

        masses of the stars: m_s.

        Returns
        -------

        J_part = np.array(3, 3*n), concatenates angular momentum for the stars,
        dark matter and gas in a single array.

        Jr_star = np.array(n, 1), Component in the plane of the angular
        momentum of stars.

        Jr = np.array(3*n, 1) Component in the plane of the angular
        momentum for all particles.

        """

        self.x_s = self.x_s - self.x_dm[p_dm.index(max(p_dm))]
        self.y_s = self.y_s - self.y_dm[p_dm.index(max(p_dm))]
        self.z_s = self.z_s - self.z_dm[p_dm.index(max(p_dm))]

        self.x_dm = self.x_dm - self.x_dm[p_dm.index(max(p_dm))]
        self.y_dm = self.y_dm - self.y_dm[p_dm.index(max(p_dm))]
        self.z_dm = self.z_dm - self.z_dm[p_dm.index(max(p_dm))]

        self.x_g = self.x_g - self.x_dm[p_dm.index(max(p_dm))]
        self.y_g = self.y_g - self.y_dm[p_dm.index(max(p_dm))]
        self.z_g = self.z_g - self.z_dm[p_dm.index(max(p_dm))]

        pos_star_rot, vel_star_rot, A = aling(self.m_s,
                                              [self.x_s, self.y_s, self.z_s],
                                              [self.vx, self.vy, self.vz],
                                              3 .R[j])

        pos_dark_rot = rot([self.x_dm, self.y_dm, self.z_dm], A)
        vel_dark_rot = rot([self.vx_dm, self.vy_dm, self.vz], A)

        pos_gas_rot = rot([self.x_g, self.y_g, self.z_g], A)
        vel_gas_rot = rot([self.vx_g, self.vy_g, self.vz_g], A)

        J_dark = np.asarray((pos_dark_rot[:, 1] * vel_dark_rot[:, 2] -
                             pos_dark_rot[:, 2] * vel_dark_rot[:, 1],
                             pos_dark_rot[:, 2] * vel_dark_rot[:, 0] -
                             pos_dark_rot[:, 0] * vel_dark_rot[:, 2],
                             pos_dark_rot[:, 0] * vel_dark_rot[:, 1] -
                             pos_dark_rot[:, 1] * vel_dark_rot[:, 0]))

        J_star = np.asarray((pos_star_rot[:, 1] * vel_star_rot[:, 2] -
                             pos_star_rot[:, 2] * vel_star_rot[:, 1],
                             pos_star_rot[:, 2] * vel_star_rot[:, 0] -
                             pos_star_rot[:, 0] * vel_star_rot[:, 2],
                             pos_star_rot[:, 0] * vel_star_rot[:, 1] -
                             pos_star_rot[:, 1] * vel_star_rot[:, 0]))

        J_gas = np.asarray((pos_gas_rot[:, 1] * vel_gas_rot[:, 2] -
                            pos_gas_rot[:, 2] * vel_gas_rot[:, 1],
                            pos_gas_rot[:, 2] * vel_gas_rot[:, 0] -
                            pos_gas_rot[:, 0] * vel_gas_rot[:, 2],
                            pos_gas_rot[:, 0] * vel_gas_rot[:, 1] -
                            pos_gas_rot[:, 1] * vel_gas_rot[:, 0]))

        J_part = np.concatenate((J_gas, J_dark, J_star), axis=1)

        Jr_star = np.sqrt(J_star[0, :]**2 + J_star[1, :]**2)

        Jr = np.sqrt(J_part[0, :]**2 + J_part[1, :]**2)

        return J_part, Jr_star, Jr
