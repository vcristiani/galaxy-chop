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
# UNIT CONVERTION
# #####################################################


@attr.s()
class UnitConverter:
    """
    Unit converter.

    This class returns a function that assigns the default units
    to any dimensionless input or if the input is a u.Quantity
    returns the units given.
    """

    default_unit = attr.ib()

    def __call__(self, v):
        """Caller function."""
        if isinstance(v, u.Quantity) and v.unit != u.dimensionless_unscaled:
            return v
        return v * self.default_unit


def unit_attribute(valid_units, default_unit, **kwargs):
    """Define unit attribute of Galaxy class."""
    if "converter" in kwargs:
        raise AttributeError("converter")
    converter = UnitConverter(default_unit=default_unit)

    def _unit_validator(instance, attribute, value):
        """Validate units."""
        if value.unit.physical_type not in valid_units:
            raise ValueError(f"Attribute '{attribute.name}'\
                              must have units of {valid_units}.\
                              Found {value.unit}")

    validators = kwargs.pop("validator", [])
    validators.append(_unit_validator)

    return attr.ib(validator=validators, converter=converter, **kwargs)


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

    x_s = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    y_s = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    z_s = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    vx_s = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    vy_s = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    vz_s = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    m_s = unit_attribute(valid_units=['mass'], default_unit=(u.Msun))

    x_dm = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    y_dm = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    z_dm = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    vx_dm = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    vy_dm = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    vz_dm = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    m_dm = unit_attribute(valid_units=['mass'], default_unit=(u.Msun))

    x_g = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    y_g = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    z_g = unit_attribute(valid_units=['length'], default_unit=(u.kpc))
    vx_g = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    vy_g = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    vz_g = unit_attribute(valid_units=['velocity'], default_unit=(u.km / u.s))
    m_g = unit_attribute(valid_units=['mass'], default_unit=(u.Msun))

    eps_s = unit_attribute(default=0,
                           valid_units=['length'], default_unit=(u.kpc))
    eps_dm = unit_attribute(default=0.,
                            valid_units=['length'], default_unit=(u.kpc))
    eps_g = unit_attribute(default=0.,
                           valid_units=['length'], default_unit=(u.kpc))

    Etot_dm = unit_attribute(default=None,
                             valid_units=['energy'],
                             default_unit=(u.Msun * (u.km / u.s)**2))
    Etot_s = unit_attribute(default=None,
                            valid_units=['energy'],
                            default_unit=(u.Msun * (u.km / u.s)**2))
    Etot_g = unit_attribute(default=None,
                            valid_units=['energy'],
                            default_unit=(u.Msun * (u.km / u.s)**2))

    # components_s = attr.ib(default=None)
    # components_g = attr.ib(default=None)
    # metadata = attr.ib(default=None)

    def energy(self):
        """
        Energy calculation.

        Calculate kinetic and potential energy of dark matter,
        star and gas particles.
        """
        x_s = self.x_s.to_value(u.kpc)
        y_s = self.y_s.to_value(u.kpc)
        z_s = self.z_s.to_value(u.kpc)

        x_g = self.x_g.to_value(u.kpc)
        y_g = self.y_g.to_value(u.kpc)
        z_g = self.z_g.to_value(u.kpc)

        x_dm = self.x_dm.to_value(u.kpc)
        y_dm = self.y_dm.to_value(u.kpc)
        z_dm = self.z_dm.to_value(u.kpc)

        m_s = self.m_s.to_value(u.Msun)
        m_g = self.m_g.to_value(u.Msun)
        m_dm = self.m_dm.to_value(u.Msun)

        eps_s = self.eps_s.to_value(u.kpc)
        eps_g = self.eps_g.to_value(u.kpc)
        eps_dm = self.eps_dm.to_value(u.kpc)

        vx_s = self.x_s.to_value(u.km / u.s)
        vy_s = self.y_s.to_value(u.km / u.s)
        vz_s = self.z_s.to_value(u.km / u.s)

        vx_g = self.x_g.to_value(u.km / u.s)
        vy_g = self.y_g.to_value(u.km / u.s)
        vz_g = self.z_g.to_value(u.km / u.s)

        vx_dm = self.x_dm.to_value(u.km / u.s)
        vy_dm = self.y_dm.to_value(u.km / u.s)
        vz_dm = self.z_dm.to_value(u.km / u.s)

        x = np.hstack((x_s, x_dm, x_g))
        y = np.hstack((y_s, y_dm, y_g))
        z = np.hstack((z_s, z_dm, z_g))
        m = np.hstack((m_s, m_dm, m_g))
        eps = np.max([eps_dm, eps_g, eps_s])

        pot = utils.potential(da.asarray(x, chunks=100),
                              da.asarray(y, chunks=100),
                              da.asarray(z, chunks=100),
                              da.asarray(m, chunks=100),
                              da.asarray(eps))

        pot_s = pot[:len(m_s)]
        pot_dm = pot[len(m_s):len(m_s) + len(m_dm)]
        pot_g = pot[len(m_s) + len(m_dm):]

        k_dm = 0.5 * (vx_dm**2 + vy_dm**2 + vz_dm**2)
        k_s = 0.5 * (vx_s**2 + vy_s**2 + vz_s**2)
        k_g = 0.5 * (vx_g**2 + vy_g**2 + vz_g**2)

        Etot_dm = k_dm - pot_dm
        Etot_s = k_s - pot_s
        Etot_g = k_g - pot_g

        setattr(self, "Etot_dm", Etot_dm * u.Msun * (u.km / u.s)**2)
        setattr(self, "Etot_s", Etot_s * u.Msun * (u.km / u.s)**2)
        setattr(self, "Etot_g", Etot_g * u.Msun * (u.km / u.s)**2)

        return Etot_dm * u.Msun * (u.km / u.s)**2,
        Etot_s * u.Msun * (u.km / u.s)**2,
        Etot_g * u.Msun * (u.km / u.s)**2
