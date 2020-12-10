# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module galaxy-chop."""

# #####################################################
# IMPORTS
# #####################################################

from astropy import units as u

import attr

import dask.array as da

from galaxychop import utils

import numpy as np

# from scipy.interpolate import InterpolatedUnivariateSpline

import uttr


# #####################################################
# CONSTANTS
# #####################################################
"""Gravitational constant G.

Units: kpc Msun**(-1) (km/s)**2"""

G = 4.299e-6

# #####################################################
# GALAXY CLASS
# #####################################################


@attr.s(frozen=True)
class Galaxy:
    """
    Galaxy class.

    Build a galaxy object from the masses, positions, and
    velocities of the particles (stars, dark matter, and gas)

    Parameters
    ----------
    x_s, y_s, z_s : `Quantity`
        Star positions. Shape: (n_s,1). Default unit: kpc.
    vx_s, vy_s, vz_s : `Quantity`
        Star velocities. Shape: (n_s,1). Default unit: km/s.
    m_s : `Quantity`
        Star masses. Shape: (n_s,1). Default unit: M_sun
    eps_s : `Quantity` Default value = 0
        Softening radius of star particles. Shape: (1,). Default unit: kpc.
    pot_s : `Quantity` Default value = 0
        Specific potential energy of star particles.
        Shape: (n_s,1). Default unit: (km/s)**2.
    x_dm, y_dm, z_dm :  `Quantity`
        Dark matter positions. Shape: (n_dm,1). Default unit: kpc.
    vx_dm, vy_dm, vz_dm : `Quantity`
        Dark matter velocities. Shape: (n_dm,1). Default unit: km/s.
    m_dm : `Quantity`
        Dark matter masses. Shape: (n_dm,1). Default unit: M_sun
    eps_dm : `Quantity` Default value = 0
        Softening radius of dark matter particles.
        Shape: (1,). Default unit: kpc.
    pot_dm : `Quantity` Default value = 0
        Specific potential energy of dark matter particles.
        Shape: (n_dm,1). Default unit: (km/s)**2.
    x_g, y_g, z_g :  `Quantity`
        Gas positions. Shape: (n_g,1). Default unit: kpc.
    vx_g, vy_g, vz_g : `Quantity`
        Gas velocities. Shape: (n_g,1). Default unit: km/s.
    m_g : `Quantity`
        Gas masses. Shape: (n_g,1). Default unit: M_sun
    eps_g : `Quantity` Default value = 0
        Softening radius of gas particles. Shape: (1,). Default unit: kpc.
    pot_g : `Quantity` Default value = 0
        Specific potential energy of gas particles.
        Shape: (n_g,1). Default unit: (km/s)**2.
    J_part : `Quantity`
        Angular momentum for gas, dark matter and stars.
        Shape: (n,3). Default units: kpc*km/s
    Jr_star : `Quantity`
        Absolute value of the angular momentum for stars.
        Shape: (n_s,1). Default unit: kpc*km/s
    Jr : `Quantity`
        Absolute value of total the angular momentum in the xy plane.
        Shape: (n,1). Default unit: kpc*km/s
    J_star : `Quantity`
        Angular momentum for stars.
        Shape: (n_s,1). Default unit: kpc*km/s
    x : `Quantity`
        Normalized energy. Default unit: (km/s)**2
    y : `Quantity`
        z component of the normalized angular momentum.
        Default units: kpc*km/s
    components_s : `np.ndarray(n_star,1)`
        Indicate the component to which the stellar particle is assigned.
        Is chosen as the most probable component.
        `n_star` is the number of stellar particles.
    components_g : `np.ndarray(n_gas,1)`
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

    pot_dm = uttr.ib(default=np.zeros(1), unit=(u.km / u.s) ** 2)
    pot_s = uttr.ib(default=np.zeros(1), unit=(u.km / u.s) ** 2)
    pot_g = uttr.ib(default=np.zeros(1), unit=(u.km / u.s) ** 2)

    J_part = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))
    Jr_star = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))
    Jr = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))
    J_star = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))

    x = uttr.ib(default=None, unit=(u.km / u.s) ** 2)
    y = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))

    arr_ = uttr.array_accessor()

    # components_s = attr.ib(default=None)
    # components_g = attr.ib(default=None)
    # metadata = attr.ib(default=None)

    def __attrs_post_init__(self):
        """
        Validate attrs with units.

        Units length validator.

        This method determine that the length of the different particles
        families are the same.

        Potential energy input validator.

        This method determine the validation of input of the specific
        potential energy.

        """
        if np.all(self.arr_.pot_s) != 0.0:
            length_s = np.array(
                [
                    len(self.arr_.y_s),
                    len(self.arr_.z_s),
                    len(self.arr_.vx_s),
                    len(self.arr_.vy_s),
                    len(self.arr_.vz_s),
                    len(self.arr_.m_s),
                    len(self.arr_.pot_s),
                ]
            )
        else:
            length_s = np.array(
                [
                    len(self.arr_.y_s),
                    len(self.arr_.z_s),
                    len(self.arr_.vx_s),
                    len(self.arr_.vy_s),
                    len(self.arr_.vz_s),
                    len(self.arr_.m_s),
                ]
            )

        if np.any(len(self.arr_.x_s) != length_s):
            raise ValueError("Stars inputs must have the same length")

        if np.all(self.arr_.pot_s) != 0.0:
            length_dm = np.array(
                [
                    len(self.arr_.y_dm),
                    len(self.arr_.z_dm),
                    len(self.arr_.vx_dm),
                    len(self.arr_.vy_dm),
                    len(self.arr_.vz_dm),
                    len(self.arr_.m_dm),
                    len(self.arr_.pot_dm),
                ]
            )
        else:
            length_dm = np.array(
                [
                    len(self.arr_.y_dm),
                    len(self.arr_.z_dm),
                    len(self.arr_.vx_dm),
                    len(self.arr_.vy_dm),
                    len(self.arr_.vz_dm),
                    len(self.arr_.m_dm),
                ]
            )

        if np.any(len(self.arr_.x_dm) != length_dm):
            raise ValueError("Dark matter inputs must have the same length")

        if np.all(self.arr_.pot_s) != 0.0:
            length_g = np.array(
                [
                    len(self.arr_.y_g),
                    len(self.arr_.z_g),
                    len(self.arr_.vx_g),
                    len(self.arr_.vy_g),
                    len(self.arr_.vz_g),
                    len(self.arr_.m_g),
                    len(self.arr_.pot_g),
                ]
            )
        else:
            length_g = np.array(
                [
                    len(self.arr_.y_g),
                    len(self.arr_.z_g),
                    len(self.arr_.vx_g),
                    len(self.arr_.vy_g),
                    len(self.arr_.vz_g),
                    len(self.arr_.m_g),
                ]
            )

        if np.any(len(self.arr_.x_g) != length_g):
            raise ValueError("Gas inputs must have the same length")

        # Potential energy input validator.
        if np.any(self.arr_.pot_dm != 0.0) and (
            np.all(self.arr_.pot_s == 0.0) or np.all(self.arr_.pot_g == 0.0)
        ):
            raise ValueError(
                "Potential energy must be instanced for all type particles"
            )

        if np.any(self.arr_.pot_s != 0.0) and (
            np.all(self.arr_.pot_dm == 0.0) or np.all(self.arr_.pot_g == 0.0)
        ):
            raise ValueError(
                "Potential energy must be instanced for all type particles"
            )

        if np.any(self.arr_.pot_g != 0.0) and (
            np.all(self.arr_.pot_s == 0.0) or np.all(self.arr_.pot_dm == 0.0)
        ):
            raise ValueError(
                "Potential energy must be instanced for all type particles"
            )

    def values(self, star=True):
        """
        2D and 1D imputs converter.

        Builds two arrays, one 2D listing all the parameters of each
        particle and one 1D showing whether the particle is a star,
        gas or dark matter.

        Parameters
        ----------
        star : bool, default=False
            Indicates if the stars particles are going to be use to
            build the array
        gas : bool, default=False
            Indicates if the gas particles are going to be use to
            build the array
        dm : bool, default=False
            Indicates if the dark matter particles are going to be use to
            build the array

        Return
        ------
        X : `np.ndarray(n,7)`
            2D array where each file it is a diferen particle and
            each column it is a parameter of the particles (E_star, eps, eps_r)
        y : `np.ndarray(n)`
            1D array where is identified the nature of each particle
            0=star, 1=gas and 2=dark matter
        """
        X = np.empty((0, 3))
        y = np.empty(0, int)

        if star:
            n_s = len(self.paramcirc[1])

            X_s = np.hstack(
                (
                    self.paramcirc[0].reshape(n_s, 1),
                    self.paramcirc[1].reshape(n_s, 1),
                    self.paramcirc[2].reshape(n_s, 1),
                )
            )
            y_s = np.zeros(n_s)

            X = np.vstack((X, X_s))
            y = np.hstack((y, y_s))

        return X, y

    @property
    def kinetic_energy(self):
        """
        Specific kinetic energy calculation.

        Calculate the specific kinetic energy
        of dark matter, star and gas particles.

        Returns
        -------
        tuple : 'Quantity'
            Specific kinetic energy of dark matter, stars and
            gas in this order.
        """
        vx_s = self.arr_.vx_s
        vy_s = self.arr_.vy_s
        vz_s = self.arr_.vz_s

        vx_g = self.arr_.vx_g
        vy_g = self.arr_.vy_g
        vz_g = self.arr_.vz_g

        vx_dm = self.arr_.vx_dm
        vy_dm = self.arr_.vy_dm
        vz_dm = self.arr_.vz_dm

        k_dm = 0.5 * (vx_dm ** 2 + vy_dm ** 2 + vz_dm ** 2)
        k_s = 0.5 * (vx_s ** 2 + vy_s ** 2 + vz_s ** 2)
        k_g = 0.5 * (vx_g ** 2 + vy_g ** 2 + vz_g ** 2)

        k_dm = k_dm * (u.km / u.s) ** 2
        k_s = k_s * (u.km / u.s) ** 2
        k_g = k_g * (u.km / u.s) ** 2

        return (k_dm, k_s, k_g)

    def potential_energy(self):
        """
        Specific potential energy calculation.

        Calculate the specific potencial energy
        of dark matter, star and gas particles.

        Returns
        -------
        gx : `galaxy object`
            New instanced galaxy specific potencial energy calculated for
            dark matter, stars and gas.
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

        num_s = len(m_s)
        num = len(m_s) + len(m_dm)

        pot_s = pot[:num_s]
        pot_dm = pot[num_s:num]
        pot_g = pot[num:]

        new = attr.asdict(self, recurse=False)
        del new["arr_"]
        new.update(
            pot_dm=-pot_dm * (u.km / u.s) ** 2,
            pot_s=-pot_s * (u.km / u.s) ** 2,
            pot_g=-pot_g * (u.km / u.s) ** 2,
        )

        return Galaxy(**new)

    @property
    def energy(self):
        """
        Specific energy calculation.

        Calculate the specific energy
        of dark matter, star and gas particles.

        Returns
        -------
        tuple : 'Quantity'
            Specific energy of dark matter, stars and gas in that order.
        """
        potential = np.concatenate(
            [
                self.arr_.pot_s,
                self.arr_.pot_dm,
                self.arr_.pot_s,
            ]
        )

        k_dm = self.kinetic_energy[0].value
        k_s = self.kinetic_energy[1].value
        k_g = self.kinetic_energy[2].value

        if np.all(potential == 0.0):
            pot_dm = self.potential_energy().arr_.pot_dm
            pot_s = self.potential_energy().arr_.pot_s
            pot_g = self.potential_energy().arr_.pot_g
        else:
            pot_dm = self.arr_.pot_dm
            pot_s = self.arr_.pot_s
            pot_g = self.arr_.pot_g

        Etot_dm = (k_dm + pot_dm) * (u.km / u.s) ** 2
        Etot_s = (k_s + pot_s) * (u.km / u.s) ** 2
        Etot_g = (k_g + pot_g) * (u.km / u.s) ** 2

        return (Etot_dm, Etot_s, Etot_g)

    def angular_momentum(self, r_cut=None):
        """
        Specific angular momentum.

        Centers the particles with respect to the one with lower specific
        potential, then calculates specific  angular momentum of
        dark matter, stars and gas particles.

        Parameters
        ----------
        r_cut : int

        Returns
        -------
        gx : `galaxy object`
            New instanced galaxy with all particles centered respect to the
            lowest specific energy one and the addition of J_part, J_star, Jr.
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

        vx_s = self.arr_.vx_s
        vy_s = self.arr_.vy_s
        vz_s = self.arr_.vz_s

        vx_g = self.arr_.vx_g
        vy_g = self.arr_.vy_g
        vz_g = self.arr_.vz_g

        vx_dm = self.arr_.vx_dm
        vy_dm = self.arr_.vy_dm
        vz_dm = self.arr_.vz_dm

        xs, ys, zs, xdm, ydm, zdm, xg, yg, zg = utils.center(
            x_s, y_s, z_s, x_dm, y_dm, z_dm, x_g, y_g, z_g, m_s, m_g, m_dm
        )

        (
            pos_rot_s_x,
            pos_rot_s_y,
            pos_rot_s_z,
            vel_rot_s_x,
            vel_rot_s_y,
            vel_rot_s_z,
            pos_rot_dm_x,
            pos_rot_dm_y,
            pos_rot_dm_z,
            vel_rot_dm_x,
            vel_rot_dm_y,
            vel_rot_dm_z,
            pos_rot_g_x,
            pos_rot_g_y,
            pos_rot_g_z,
            vel_rot_g_x,
            vel_rot_g_y,
            vel_rot_g_z,
        ) = utils.align(
            m_s,
            xs,
            ys,
            zs,
            vx_s,
            vy_s,
            vz_s,
            xdm,
            ydm,
            zdm,
            vx_dm,
            vy_dm,
            vz_dm,
            xg,
            yg,
            zg,
            vx_g,
            vy_g,
            vz_g,
            r_cut=r_cut,
        )

        J_dark = np.array(
            [
                pos_rot_dm_y * vel_rot_dm_z - pos_rot_dm_z * vel_rot_dm_y,
                pos_rot_dm_z * vel_rot_dm_x - pos_rot_dm_x * vel_rot_dm_z,
                pos_rot_dm_x * vel_rot_dm_y - pos_rot_dm_y * vel_rot_dm_x,
            ]
        )

        J_star = np.array(
            [
                pos_rot_s_y * vel_rot_s_z - pos_rot_s_z * vel_rot_s_y,
                pos_rot_s_z * vel_rot_s_x - pos_rot_s_x * vel_rot_s_z,
                pos_rot_s_x * vel_rot_s_y - pos_rot_s_y * vel_rot_s_x,
            ]
        )

        J_gas = np.array(
            [
                pos_rot_g_y * vel_rot_g_z - pos_rot_g_z * vel_rot_g_y,
                pos_rot_g_z * vel_rot_g_x - pos_rot_g_x * vel_rot_g_z,
                pos_rot_g_x * vel_rot_g_y - pos_rot_g_y * vel_rot_g_x,
            ]
        )

        J_part = np.concatenate([J_gas, J_dark, J_star], axis=1)

        Jr_star = np.sqrt(J_star[0, :] ** 2 + J_star[1, :] ** 2)

        Jr = np.sqrt(J_part[0, :] ** 2 + J_part[1, :] ** 2)

        new = attr.asdict(self, recurse=False)
        del new["arr_"]
        new.update(
            J_part=J_part * u.kpc * u.km / u.s,
            Jr_star=Jr_star * u.kpc * u.km / u.s,
            Jr=Jr * u.kpc * u.km / u.s,
            J_star=J_star * u.kpc * u.km / u.s,
        )

        return Galaxy(**new)

    def jcirc(self, bin0=0.05, bin1=0.005):
        """
        Circular angular momentum.

        Calculation of the dots to build the function of the circular
        angular momentum.

        Parameters
        ----------
        bin0 : float, default=0.05
            Size of the specific energy bin of the inner part of the galaxy,
            in the range of (-1, -0.1) of the normalized energy.
        bin1 : float, default=0.005
            Size of the specific energy bin of the outer part of the galaxy,
            in the range of (-0.1, 0) of the normalized energy.

        Returns
        -------
        gx : `galaxy object`
            New instanced galaxy with x (normalized specific energy) and
            y (z component of the normalized specific angular momentum).
            See section Notes for more details.

        Notes
        -----
            The x and y values are calculated from the binning in the
            normalized specific energy. In each bin, the particle with the
            highest value of z component of standardized specific angular
            momentum is selected, and its value of normalized specific energy
            is assigned to x and its value of the z component of the normalized
            specific angular momentum to y.
        """
        Etot_dm = self.energy[0].value
        Etot_s = self.energy[1].value
        Etot_g = self.energy[2].value

        E_tot = np.hstack([Etot_s, Etot_dm, Etot_g])

        # Remove the particles that are not bound: E > 0.
        (neg,) = np.where(E_tot <= 0.0)
        (neg_star,) = np.where(Etot_s <= 0.0)

        # Remove the particles with E = -inf.
        (fin,) = np.where(E_tot[neg] != -np.inf)
        (fin_star,) = np.where(Etot_s[neg_star] != -np.inf)

        # Normalize the two variables: E between 0 and 1; Jz between -1 and 1.
        E = E_tot[neg][fin] / np.abs(np.min(E_tot[neg][fin]))

        kk = self.angular_momentum().arr_.J_part[2, :][neg][fin]

        Jz = kk / np.max(np.abs(kk))

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
                        1.0
                        - (np.abs(Jz[mask][s][-2]) / np.abs(Jz[mask][s][-1]))
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

        if len(mask) != 0:
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

        new = attr.asdict(self, recurse=False)
        del new["arr_"]
        new.update(x=x * (u.km / u.s) ** 2, y=y * u.kpc * u.km / u.s)

        return Galaxy(**new)

    @property
    def paramcirc(self):
        """
        Circular parameters calculation.

        Return
        ------
        tuple : Quantity
            Normalized specific energy of the stars, J_z/J_circ, J_p/J_circ.

        Notes
        -----
        J_z : z-component of normalized specific angular momentum.
        J_circ : circular specific angular momentum.
        J_p : module of the projection on the xy plane of the normalized
        specific angular momentum.
        """
        Etot_dm = self.energy[0].value
        Etot_s = self.energy[1].value
        Etot_g = self.energy[2].value

        E_tot = np.hstack([Etot_s, Etot_dm, Etot_g])

        # Remove the particles that are not bound: E > 0.
        (neg,) = np.where(E_tot <= 0.0)
        (neg_star,) = np.where(Etot_s <= 0.0)

        # Remove the particles with E = -inf.
        (fin,) = np.where(E_tot[neg] != -np.inf)
        (fin_star,) = np.where(Etot_s[neg_star] != -np.inf)

        # Normalize E, Lz and Lr for the stars.
        up1 = Etot_s[neg_star][fin_star]
        down1 = np.abs(np.min(E_tot[neg][fin]))
        E_star = up1 / down1

        ang_momentum = self.angular_momentum().arr_
        up2 = ang_momentum.J_star[2, :][neg_star][fin_star]
        down2 = np.max(np.abs(ang_momentum.J_part[2, :][neg][fin]))

        Jz_star_norm = up2 / down2

        up3 = ang_momentum.Jr_star[neg_star][fin_star]
        down3 = np.max(np.abs(ang_momentum.Jr[neg][fin]))
        Jr_star_norm = up3 / down3

        # We do the interpolation to calculate the J_circ.
        # spl = InterpolatedUnivariateSpline(
        #    self.jcirc().arr_.x,
        #    self.jcirc().arr_.y,
        #    k=1,
        # )

        # Calculate the circularity parameter Lz/Lc.
        # eps = J_star_ / spl(E_star)
        jcir = self.jcirc().arr_
        eps = Jz_star_norm / np.interp(E_star, jcir.x, jcir.y)

        # Calculate the same for Lp/Lc.
        # eps_r = Jr_star_ / spl(E_star)
        eps_r = Jr_star_norm / np.interp(E_star, jcir.x, jcir.y)

        # We remove particles that have circularity < -1 and circularity > 1.
        (mask,) = np.where((eps <= 1.0) & (eps >= -1.0))

        E_star_ = u.Quantity(E_star[mask])
        eps_ = u.Quantity(eps[mask])
        eps_r_ = u.Quantity(eps_r[mask])

        return E_star_, eps_, eps_r_
