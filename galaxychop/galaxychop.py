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
from scipy.interpolate import InterpolatedUnivariateSpline

# from sklearn.mixture import GaussianMixture
# import random

import dask.array as da

# #####################################################
# CONSTANTS
# #####################################################

G = 4.299e4

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

    x = attr.ib(default=None)
    y = attr.ib(default=None)

    components_s = attr.ib(default=None)
    components_g = attr.ib(default=None)
    metadata = attr.ib(default=None)

    def energy(self, eps=0):
        """Calculation of kinetic and potencial energy of
        dark matter, star and gas particles"""

        x = np.hstack((self.x_s, self.x_dm, self.x_g))
        y = np.hstack((self.y_s, self.y_dm, self.y_g))
        z = np.hstack((self.z_s, self.z_dm, self.z_g))
        m = np.hstack((self.m_s, self.m_dm, self.m_g))

        a = utils.potential_dask(
            da.asarray(x, chunks=100),
            da.asarray(y, chunks=100),
            da.asarray(z, chunks=100),
            da.asarray(m, chunks=100),
            da.asarray(eps),
        )

        pot = a.compute()

        pot_star = pot[:len(self.m_s)]
        pot_dark = pot[len(self.m_s):len(self.m_s) + len(self.m_dm)]
        pot_gas = pot[len(self.m_s) + len(self.m_dm):]

        k_dm = 0.5 * (self.vx_dm ** 2 + self.vy_dm ** 2 + self.vz_dm ** 2)
        k_s = 0.5 * (self.vx_s ** 2 + self.vy_s ** 2 + self.vz_s ** 2)
        k_g = 0.5 * (self.vx_g ** 2 + self.vy_g ** 2 + self.vz_g ** 2)

        E_tot_dark = k_dm - pot_dark
        E_tot_star = k_s - pot_star
        E_tot_gas = k_g - pot_gas

        setattr(self, "E_tot_dark", E_tot_dark)
        setattr(self, "E_tot_star", E_tot_star)
        setattr(self, "E_tot_gas", E_tot_gas)

        return E_tot_star, E_tot_dark, E_tot_gas

    def jcirc(self, bin0=0.05, bin1=0.005):

        if np.all(getattr(self, "E_tot_dark")) == (None):
            self.energy()

        E_tot = np.hstack((self.E_tot_star, self.E_tot_dark, self.E_tot_gas))

        # Remove the particles that are not bound: E > 0.
        (neg,) = np.where(E_tot <= 0.0)
        (neg_star,) = np.where(self.E_tot_star <= 0.0)

        # Remove the particles with E = -inf.
        (fin,) = np.where(E_tot[neg] != -np.inf)
        (fin_star,) = np.where(self.E_tot_star[neg_star] != -np.inf)

        # Normalize the two variables: E between 0 and 1; J between -1 and 1.
        E = E_tot[neg][fin] / np.abs(np.min(E_tot[neg][fin]))
        J = self.J_part[2, :][neg][fin] / np.max(
            np.abs(self.J_part[2, :][neg][fin])
        )

        # Make the energy binning and select the Jz values with which we
        # calculate the J_circ.
        aux0 = np.arange(-1.0, -0.1, bin0)
        aux1 = np.arange(-0.1, 0.0, bin1)

        aux = np.concatenate((aux0, aux1), axis=0)

        x = np.zeros(len(aux) + 1)
        y = np.zeros(len(aux) + 1)

        x[0] = -1.0
        y[0] = np.abs(J[np.argmin(E)])

        for i in range(1, len(aux)):
            (mask,) = np.where((E <= aux[i]) & (E > aux[i - 1]))
            s = np.argsort(np.abs(J[mask]))

            # We take into account whether or not there are particles in the
            # energy bins.
            if len(s) != 0:
                if len(s) == 1:
                    x[i] = E[mask][s]
                    y[i] = np.abs(J[mask][s])
                else:
                    if (
                        1.0 - (np.abs(J[mask][s][-2]) / np.abs(J[mask][s][-1]))
                    ) >= 0.01:
                        x[i] = E[mask][s][-2]
                        y[i] = np.abs(J[mask][s][-2])
                    else:
                        x[i] = E[mask][s][-1]
                        y[i] = np.abs(J[mask][s][-1])
            else:
                pass

        # Mask to complete the last bin, in case there are no empty bins.
        (mask,) = np.where(E > aux[len(aux) - 1])

        if len(mask) != 0:
            x[len(aux)] = E[mask][np.abs(J[mask]).argmax()]
            y[len(aux)] = np.abs(J[mask][np.abs(J[mask]).argmax()])

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

        setattr(self, "x", x)
        setattr(self, "y", y)

        return x, y

    def paramcirc(self):

        if np.all(getattr(self, "E_tot_dark")) == (None):
            self.energy()

        if np.all(getattr(self, "x")) == (None):
            self.jcirc()

        E_tot = np.hstack((self.E_tot_star, self.E_tot_dark, self.E_tot_gas))

        # Remove the particles that are not bound: E > 0.
        (neg,) = np.where(E_tot <= 0.0)
        (neg_star,) = np.where(self.E_tot_star <= 0.0)

        # Remove the particles with E = -inf.
        (fin,) = np.where(E_tot[neg] != -np.inf)
        (fin_star,) = np.where(self.E_tot_star[neg_star] != -np.inf)

        # Normalize E, Lz and Lr for the stars.
        E_star = self.E_tot_star[neg_star][fin_star] / np.abs(
            np.min(E_tot[neg][fin])
        )

        J_star_ = self.J_star[2, :][neg_star][fin_star] / np.max(
            np.abs(self.J_part[2, :][neg][fin])
        )

        Jr_star_ = self.Jr_star[neg_star][fin_star] / np.max(
            np.abs(self.Jr[neg][fin])
        )

        # We do the interpolation to calculate the J_circ.
        spl = InterpolatedUnivariateSpline(self.x, self.y, k=1)

        # Calculate the circularity parameter Lz/Lc.
        eps = J_star_ / spl(E_star)

        # Calculate the same for Lp/Lc.
        eps_r = Jr_star_ / spl(E_star)

        # Determine that the particles we are removing are not significant.

        # We remove particles that have circularity < -1 and circularity > 1.
        (mask,) = np.where((eps <= 1.0) & (eps >= -1.0))

        # ID of the clean stellar particles.
        # ID_star = np.arange(0, len(star))[neg_star][fin_star][mask]

        # Save ID, E and circularity of the stellar particles we are
        # interested in.
        # indice = np.arange(0, len(ID_star))
        # stack = np.column_stack((ID_star, E_star[mask], eps[mask],
        #                         eps_r[mask], indice))
        stack = np.column_stack((E_star[mask], eps[mask], eps_r[mask]))
        return stack
