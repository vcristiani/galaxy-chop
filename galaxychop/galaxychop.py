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
from scipy.interpolate import InterpolatedUnivariateSpline

# from sklearn.mixture import GaussianMixture
# import random

# #####################################################
# CONSTANTS
# #####################################################
"""Gravitational constant G.

Units: kpc Msun**(-1) (km/s)**2"""

G = 4.299e-6

# #####################################################
# GALAXY CLASS
# #####################################################


@attr.s(frozen=False)
class Galaxy:
    """
    Galaxy class.

    Build a galaxy object from the masses, positions, and
    velocities of the particles (stars, dark matter, and gas)

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

    eps_s = uttr.ib(default=0.0, unit=u.kpc)
    eps_dm = uttr.ib(default=0.0, unit=u.kpc)
    eps_g = uttr.ib(default=0.0, unit=u.kpc)

    J_part = uttr.ib(default=0.0, unit=u.kpc * u.km / u.s)
    Jr_star = uttr.ib(default=0.0, unit=u.kpc * u.km / u.s)
    Jr = uttr.ib(default=0.0, unit=u.kpc * u.km / u.s)
    J_star = uttr.ib(default=0.0, unit=u.kpc * u.km / u.s)

    x = uttr.ib(default=0.0, unit=(u.km / u.s) ** 2)
    y = uttr.ib(default=0.0, unit=u.kpc * u.km / u.s)

    arr_ = uttr.array_accessor()

    # components_s = attr.ib(default=None)
    # components_g = attr.ib(default=None)
    # metadata = attr.ib(default=None)

    @property
    def energy(self):
        """
        Energy calculation.

        Calculate kinetic and potencial energy of dark matter,
        star and gas particles.
        """
        x_s = self.arr_.x_s
        y_s = self.arr_.y_s
        z_s = self.arr_.z_s

        x_g = self.arr_.x_g
        y_g = self.arr_.y_g
        z_g = self.arr_.z_g

        x_dm = self.arr_.x_dm
        y_dm = self.arr_.y_dm
        z_dm = self.arr_.z_dm

        m_s = self.arr_.m_s
        m_g = self.arr_.m_g
        m_dm = self.arr_.m_dm

        eps_s = self.arr_.eps_s
        eps_g = self.arr_.eps_g
        eps_dm = self.arr_.eps_dm

        vx_s = self.arr_.x_s
        vy_s = self.arr_.y_s
        vz_s = self.arr_.z_s

        vx_g = self.arr_.x_g
        vy_g = self.arr_.y_g
        vz_g = self.arr_.z_g

        vx_dm = self.arr_.x_dm
        vy_dm = self.arr_.y_dm
        vz_dm = self.arr_.z_dm

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

        pot_s = pot[: len(m_s)]
        pot_dm = pot[len(m_s):len(m_s) + len(m_dm)]
        pot_g = pot[len(m_s) + len(m_dm):]

        k_dm = 0.5 * (vx_dm ** 2 + vy_dm ** 2 + vz_dm ** 2)
        k_s = 0.5 * (vx_s ** 2 + vy_s ** 2 + vz_s ** 2)
        k_g = 0.5 * (vx_g ** 2 + vy_g ** 2 + vz_g ** 2)

        Etot_dm = k_dm - pot_dm
        Etot_s = k_s - pot_s
        Etot_g = k_g - pot_g

        return (Etot_dm * (u.km / u.s) ** 2,
                Etot_s * (u.km / u.s) ** 2,
                Etot_g * (u.km / u.s) ** 2)

    def angular_momentum(self, r_corte=None):
        """
        Angular Momentum.

        Centers the particles with respect to the one with lower potential
        then calculates angular momentum of dark matter, star and
        gas particles.
        """
        x_s = self.arr_.x_s
        y_s = self.arr_.y_s
        z_s = self.arr_.z_s

        x_g = self.arr_.x_g
        y_g = self.arr_.y_g
        z_g = self.arr_.z_g

        x_dm = self.arr_.x_dm
        y_dm = self.arr_.y_dm
        z_dm = self.arr_.z_dm

        m_s = self.arr_.m_s
        m_g = self.arr_.m_g
        m_dm = self.arr_.m_dm

        vx_s = self.arr_.x_s
        vy_s = self.arr_.y_s
        vz_s = self.arr_.z_s

        vx_g = self.arr_.x_g
        vy_g = self.arr_.y_g
        vz_g = self.arr_.z_g

        vx_dm = self.arr_.x_dm
        vy_dm = self.arr_.y_dm
        vz_dm = self.arr_.z_dm

        (
            xs, ys, zs,
            xdm, ydm, zdm,
            xg, yg, zg) = utils._center(x_s, y_s, z_s,
                                        x_dm, y_dm, z_dm,
                                        x_g, y_g, z_g,
                                        m_s, m_g, m_dm)

        (
            pos_rot_s_x, pos_rot_s_y, pos_rot_s_z,
            vel_rot_s_x, vel_rot_s_y, vel_rot_s_z,
            pos_rot_dm_x, pos_rot_dm_y, pos_rot_dm_z,
            vel_rot_dm_x, vel_rot_dm_y, vel_rot_dm_z,
            pos_rot_g_x, pos_rot_g_y, pos_rot_g_z,
            vel_rot_g_x, vel_rot_g_y, vel_rot_g_z,
            ) = utils.aling(m_s, xs, ys, zs,
                            vx_s, vy_s, vz_s,
                            xdm, ydm, zdm,
                            vx_dm, vy_dm, vz_dm,
                            xg, yg, zg,
                            vx_g, vy_g, vz_g,
                            r_corte=5)

        J_dark = np.array(
            [pos_rot_dm_y * vel_rot_dm_z - pos_rot_dm_z * vel_rot_dm_y,
             pos_rot_dm_z * vel_rot_dm_x - pos_rot_dm_x * vel_rot_dm_z,
             pos_rot_dm_x * vel_rot_dm_y - pos_rot_dm_y * vel_rot_dm_x])

        J_star = np.array(
            [pos_rot_s_y * vel_rot_s_z - pos_rot_s_z * vel_rot_s_y,
             pos_rot_s_z * vel_rot_s_x - pos_rot_s_x * vel_rot_s_z,
             pos_rot_s_x * vel_rot_s_y - pos_rot_s_y * vel_rot_s_x])

        J_gas = np.array(
            [pos_rot_g_y * vel_rot_g_z - pos_rot_g_z * vel_rot_g_y,
             pos_rot_g_z * vel_rot_g_x - pos_rot_g_x * vel_rot_g_z,
             pos_rot_g_x * vel_rot_g_y - pos_rot_g_y * vel_rot_g_x])

        J_part = np.concatenate([J_gas, J_dark, J_star], axis=1)

        Jr_star = np.sqrt(J_star[0, :] ** 2 + J_star[1, :] ** 2)

        Jr = np.sqrt(J_part[0, :] ** 2 + J_part[1, :] ** 2)

        new = attr.asdict(self, recurse=False)
        del new["arr_"]
        new.update(
            J_part=J_part * u.kpc * u.km / u.s,
            Jr_star=Jr_star * u.kpc * u.km / u.s,
            Jr=Jr * u.kpc * u.km / u.s,
            J_star=J_star * u.kpc * u.km / u.s
        )

        return Galaxy(**new)

    def jcirc(self, bin0=0.05, bin1=0.005):
        """
        Circular angular momentum.

        Calculation of the dots to build the function of the circular
        angular momentum.
        """
        E_tot = np.hstack((
            self.energy[1].value,
            self.energy[0].value,
            self.energy[2].value
        ))

        # Remove the particles that are not bound: E > 0.
        (neg,) = np.where(E_tot <= 0.0)
        (neg_star,) = np.where(self.energy[1].value <= 0.0)

        # Remove the particles with E = -inf.
        (fin,) = np.where(E_tot[neg] != -np.inf)
        (fin_star,) = np.where(self.energy[1].value[neg_star] != -np.inf)

        # Normalize the two variables: E between 0 and 1; J between -1 and 1.
        E = E_tot[neg][fin] / np.abs(np.min(E_tot[neg][fin]))

        up = self.angular_momentum().arr_.J_part[2, :][neg][fin]

        down = np.max(
            np.abs(self.angular_momentum().arr_.J_part[2, :][neg][fin])
        )

        J = up / down

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

        new = attr.asdict(self, recurse=False)
        del new["arr_"]
        new.update(x=x * (u.km / u.s) ** 2, y=y * u.kpc * u.km / u.s)

        return Galaxy(**new)

    @property
    def paramcirc(self):
        """Circular parameters calculation."""
        E_tot = np.hstack((
            self.energy[1].value,
            self.energy[0].value,
            self.energy[2].value
        ))

        # Remove the particles that are not bound: E > 0.
        (neg,) = np.where(E_tot <= 0.0)
        (neg_star,) = np.where(self.energy[1].value <= 0.0)

        # Remove the particles with E = -inf.
        (fin,) = np.where(E_tot[neg] != -np.inf)
        (fin_star,) = np.where(self.energy[1].value[neg_star] != -np.inf)

        # Normalize E, Lz and Lr for the stars.
        up1 = self.energy[1].value[neg_star][fin_star]
        down1 = np.abs(np.min(E_tot[neg][fin]))
        E_star = up1 / down1

        up2 = self.angular_momentum().arr_.J_star[2, :][neg_star][fin_star]
        down2 = np.max(
            np.abs(self.angular_momentum().arr_.J_part[2, :][neg][fin])
        )

        J_star_ = up2 / down2

        up3 = self.angular_momentum().arr_.Jr_star[neg_star][fin_star]
        down3 = np.max(np.abs(self.angular_momentum().arr_.Jr[neg][fin]))
        Jr_star_ = up3 / down3

        # We do the interpolation to calculate the J_circ.
        spl = InterpolatedUnivariateSpline(self.jcirc().arr_.x,
                                           self.jcirc().arr_.y,
                                           k=1)

        # Calculate the circularity parameter Lz/Lc.
        eps = J_star_ / spl(E_star)

        # Calculate the same for Lp/Lc.
        eps_r = Jr_star_ / spl(E_star)

        # Determine that the particles we are removing are not significant.

        # We remove particles that have circularity < -1 and circularity > 1.
        (mask,) = np.where((eps <= 1.0) & (eps >= -1.0))

        E_star = E_star[mask]
        eps = eps[mask]
        eps_r = eps_r[mask]

        return (E_star * (u.km / u.s) ** 2, eps, eps_r)
