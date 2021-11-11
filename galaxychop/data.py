# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Module galaxy-chop."""

# =============================================================================
# IMPORTS
# =============================================================================

import enum
from collections import defaultdict

from astropy import units as u

import attr

import numpy as np

import pandas as pd

import uttr


# =============================================================================
# PARTICLE SET
# =============================================================================
class ParticleSetType(enum.IntEnum):
    """
    Name of the particle type.

    Name and number that are used to describe the particle
    type in the ParticleSet class.
    """

    STARS = 0
    DARK_MATTER = 1
    GAS = 2

    @classmethod
    def mktype(cls, v):
        """Creates a ParticleSetType from name, or value."""
        if isinstance(v, ParticleSetType):
            return v
        if isinstance(v, str):
            v = v.upper()
        for p in ParticleSetType:
            if v in (p.name, p.value):
                return p
        raise ValueError(f"Can't coherce {v} into ParticleSetType ")

    def humanize(self):
        return self.name.lower()


@uttr.s(frozen=True, slots=True, repr=False)
class ParticleSet:
    """
    ParticleSet class.

    Creates a set particles of a particular type (stars, dark matter or gas)
    using masses, positions, velocities and potential energy.

    Parameters
    ----------
    ptype :
    m : `Quantity`
        Masses. Shape: (n,1). Default unit: M_sun
    x, y, z : `Quantity`
        Positions. Shape: (n,1). Default unit: kpc.
    vx, vy, vz : `Quantity`
        Velocities. Shape: (n,1). Default unit: km/s.
    potential : `Quantity`, default value = 0
        Specific potential energy of particles.
        Shape: (n,1). Default unit: (km/s)**2.
    softening : `Quantity`, default value = 0
        Softening radius of particles. Shape: (1,). Default unit: kpc.
    kinetic_energy : `Quantity`
        Specific kinetic energy of particles.
        Shape: (n,1). Default unit: (km/s)**2.
    total_energy : `Quantity`
        Specific total energy of particles.
        Shape: (n,1). Default unit: (km/s)**2.
    Jx, Jy, Jz : `Quantity`
        Components of angular momentum of particles.
        Shape: (n,3). Default units: kpc*km/s.

    Attributes
    ----------
    arr_: `uttr.ArrayAccessor`
        Original array accessor object create by the *uttr* library.
        Array accesor: it converts uttr attributes to the default unit and
        afterward to a `numpy.ndarray`.
        For more information see: https://pypi.org/project/uttrs/
    """

    ptype = uttr.ib(validator=attr.validators.instance_of(ParticleSetType))

    m: np.ndarray = uttr.ib(unit=u.Msun)
    x: np.ndarray = uttr.ib(unit=u.kpc)
    y: np.ndarray = uttr.ib(unit=u.kpc)
    z: np.ndarray = uttr.ib(unit=u.kpc)
    vx: np.ndarray = uttr.ib(unit=(u.km / u.s))
    vy: np.ndarray = uttr.ib(unit=(u.km / u.s))
    vz: np.ndarray = uttr.ib(unit=(u.km / u.s))

    potential: np.ndarray = uttr.ib(
        unit=(u.km / u.s) ** 2,
        validator=attr.validators.optional(
            attr.validators.instance_of(np.ndarray)
        ),
        repr=False,
    )

    softening: float = uttr.ib(converter=float, repr=False)

    has_potential_: bool = uttr.ib(init=False)
    kinetic_energy_: np.ndarray = uttr.ib(unit=(u.km / u.s) ** 2, init=False)
    total_energy_: np.ndarray = uttr.ib(unit=(u.km / u.s) ** 2, init=False)

    # angular momentum
    Jx_ = uttr.ib(unit=(u.kpc * u.km / u.s), init=False)
    Jy_ = uttr.ib(unit=(u.kpc * u.km / u.s), init=False)
    Jz_ = uttr.ib(unit=(u.kpc * u.km / u.s), init=False)

    # UTTRS Orchestration =====================================================

    @has_potential_.default
    def _has_potential__default(self):
        return self.potential is not None

    @kinetic_energy_.default
    def _kinetic_energy__default(self):
        arr = self.arr_
        ke = 0.5 * (arr.vx ** 2 + arr.vy ** 2 + arr.vz ** 2)
        return ke

    @total_energy_.default
    def _total_energy__default(self):
        if not self.has_potential_:
            return

        arr = self.arr_
        kenergy = arr.kinetic_energy_
        penergy = arr.potential

        return kenergy + penergy

    # angular momentum
    @Jx_.default
    def _Jx__default(self):
        arr = self.arr_
        return arr.y * arr.vz - arr.z * arr.vy  # x

    @Jy_.default
    def _Jy__default(self):
        arr = self.arr_
        return arr.z * arr.vx - arr.x * arr.vz  # y

    @Jz_.default
    def _Jz__default(self):
        arr = self.arr_
        return arr.x * arr.vy - arr.y * arr.vx  # z

    def __attrs_post_init__(self):
        """
        Particle sets length validator.

        This method determines that the length of the different particle
        attributes are the same families are the same.

        """
        # we create a dictionary where we are going to put the length as keys,
        # and the name of component with this length inside a set.
        lengths = defaultdict(set)

        lengths[len(self.m)].add("m")
        lengths[len(self.x)].add("x")
        lengths[len(self.y)].add("y")
        lengths[len(self.z)].add("z")
        lengths[len(self.vx)].add("vx")
        lengths[len(self.vy)].add("vy")
        lengths[len(self.vz)].add("vz")

        if self.has_potential_:
            lengths[len(self.potential)].add("potential")

        # now if we have more than one key it is because there are
        # different lengths.
        if len(lengths) > 1:
            raise ValueError(
                f"{self.ptype} inputs must have the same length. "
                f"Lengths: {lengths}"
            )

    # PROPERTIES ==============================================================

    @property
    def angular_momentum_(self):
        arr = self.arr_
        return np.array([arr.Jx_, arr.Jy_, arr.Jz_]) * (u.kpc * u.km / u.s)

    # REDEFINITIONS ===========================================================

    def __repr__(self):
        return (
            f"ParticleSet({self.ptype.name}, size={len(self)}, "
            f"softening={self.softening}, potentials={self.has_potential_})"
        )

    def __len__(self):
        return len(self.m)

    # UTILITIES ===============================================================

    def to_dataframe(self, attributes=None):
        """
        Data frame builder

        This method builds a data frame of all parameters of ParticleSet class.

        Parameters
        ----------
        attributes: tuple, default value = None
            Dictionary keys of all ParticleSet class parameters.

        Return
        ------
        DataFrame : pandas data frame
            Data frame of all ParticleSet class parameters
            of the particles.

        """
        arr = self.arr_
        columns_makers = {
            "ptype": lambda: np.full(len(self), self.ptype.humanize()),
            "ptypev": lambda: np.full(len(self), self.ptype.value),
            "m": lambda: arr.m,
            "x": lambda: arr.x,
            "y": lambda: arr.y,
            "z": lambda: arr.z,
            "vx": lambda: arr.vx,
            "vy": lambda: arr.vy,
            "vz": lambda: arr.vz,
            "softening": lambda: self.softening,
            "potential": lambda: (
                arr.potential if self.has_potential_ else np.nan
            ),
            "kinetic_energy": lambda: arr.kinetic_energy_,
            "total_energy": lambda: (
                arr.total_energy_ if self.has_potential_ else np.nan
            ),
            "Jx": lambda: arr.Jx_,
            "Jy": lambda: arr.Jy_,
            "Jz": lambda: arr.Jz_,
        }
        attributes = (
            columns_makers.keys() if attributes is None else attributes
        )
        data = {}
        for aname in attributes:
            mkcolumn = columns_makers[aname]
            data[aname] = mkcolumn()
        return pd.DataFrame(data)


# =============================================================================
# GALAXY CLASS
# =============================================================================


@uttr.s(frozen=True)
class Galaxy:
    """
    Galaxy class.

    Builds a galaxy object from a ParticleSet for each type
    of particle (stars, dark matter and gas).
    """

    stars = uttr.ib(validator=attr.validators.instance_of(ParticleSet))
    dark_matter = uttr.ib(validator=attr.validators.instance_of(ParticleSet))
    gas = uttr.ib(validator=attr.validators.instance_of(ParticleSet))

    has_potential_: bool = attr.ib(init=False)

    @has_potential_.default
    def _has_potential__default(self):
        # this is a set only can have 3 possible values:
        # 1. {True} all the components has potential
        # 2. {False} No component has potential
        # 3. {True, False} mixed <- This is an error
        has_pot = {
            "stars": self.stars.has_potential_,
            "gas": self.gas.has_potential_,
            "dark_matter": self.dark_matter.has_potential_,
        }
        if set(has_pot.values()) == {True, False}:
            raise ValueError(
                "Potential energy must be instanced for all particles types. "
                f"Found: {has_pot}"
            )
        return self.stars.has_potential_

    def __attrs_post_init__(self):
        """Validate that the type of each particleset is correct."""
        pset_types = [
            ("stars", self.stars, ParticleSetType.STARS),
            ("dark_matter", self.dark_matter, ParticleSetType.DARK_MATTER),
            ("gas", self.gas, ParticleSetType.GAS),
        ]
        for psname, pset, pstype in pset_types:
            if pset.ptype != pstype:
                raise TypeError(f"{psname} must be of type {pstype}")

    def __len__(self):
        return len(self.stars) + len(self.dark_matter) + len(self.gas)

    # UTILITIES ===============================================================

    def to_dataframe(self, *, ptypes=None, attributes=None):
        psets = [self.stars, self.dark_matter, self.gas]

        parts = []
        for pset in psets:
            if ptypes is None or pset.ptype.humanize() in ptypes:
                df = pset.to_dataframe(attributes=attributes)
                parts.append(df)

        return pd.concat(parts, ignore_index=True)

    @property
    def plot(self):
        """Plot accessor."""
        from . import plot  # noqa

        return plot.GalaxyPlotter(self)

    # ENERGY ===============================================================

    @property
    def kinetic_energy_(self):
        """Specific kinetic energy of stars, dark matter and gas particles.

        Returns
        -------
        tuple : `Quantity`
            (k_s, k_dm, k_g): Specific kinetic energy of stars, dark matter and
            gas respectively. Shape(n_s, n_dm, n_g). Unit: (km/s)**2

        Examples
        --------
        This returns the specific kinetic energy of stars, dark matter and gas
        particles respectively.

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> k_s, k_dm, k_g = galaxy.kinetic_energy
        """
        return (
            self.stars.kinetic_energy_,
            self.dark_matter.kinetic_energy_,
            self.gas.kinetic_energy_,
        )

    @property
    def potential_energy_(self):
        if self.has_potential_:
            return (
                self.stars.potential,
                self.dark_matter.potential,
                self.gas.potential,
            )

    @property
    def total_energy_(self):
        """
        Specific energy calculation.

        Calculates the specific energy of dark matter, star and gas particles.

        Returns
        -------
        tuple : `Quantity`
            (Etot_s, Etot_dm, Etot_g): Specific total energy of stars, dark
            matter and gas respectively.
            Shape(n_s, n_dm, n_g). Unit: (km/s)**2

        Examples
        --------
        This returns the specific total energy of stars, dark matter and gas
        particles respectively.

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> E_s, E_dm, E_g = galaxy.energy
        """
        if self.has_potential_:
            return (
                self.stars.total_energy_,
                self.dark_matter.total_energy_,
                self.gas.total_energy_,
            )

    @property
    def angular_momentum_(self):
        return (
            self.stars.angular_momentum_,
            self.dark_matter.angular_momentum_,
            self.gas.angular_momentum_,
        )


# =============================================================================
# API FUNCTIONS
# =============================================================================


def galaxy_as_kwargs(galaxy):
    def _filter_internals(attribute, value):
        return attribute.init

    def _pset_as_kwargs(pset, suffix):
        return {f"{k}_{suffix}": v for k, v in pset.items() if k != "ptype"}

    gkwargs = attr.asdict(galaxy, recurse=True, filter=_filter_internals)

    stars_kws = _pset_as_kwargs(gkwargs.pop("stars"), "s")
    dark_matter_kws = _pset_as_kwargs(gkwargs.pop("dark_matter"), "dm")
    gas_kws = _pset_as_kwargs(gkwargs.pop("gas"), "g")

    gkwargs.update(**stars_kws, **dark_matter_kws, **gas_kws)

    return gkwargs


def mkgalaxy(
    m_s: np.ndarray,
    x_s: np.ndarray,
    y_s: np.ndarray,
    z_s: np.ndarray,
    vx_s: np.ndarray,
    vy_s: np.ndarray,
    vz_s: np.ndarray,
    m_dm: np.ndarray,
    x_dm: np.ndarray,
    y_dm: np.ndarray,
    z_dm: np.ndarray,
    vx_dm: np.ndarray,
    vy_dm: np.ndarray,
    vz_dm: np.ndarray,
    m_g: np.ndarray,
    x_g: np.ndarray,
    y_g: np.ndarray,
    z_g: np.ndarray,
    vx_g: np.ndarray,
    vy_g: np.ndarray,
    vz_g: np.ndarray,
    softening_s: float = 0.0,
    softening_dm: float = 0.0,
    softening_g: float = 0.0,
    potential_s: np.ndarray = None,
    potential_dm: np.ndarray = None,
    potential_g: np.ndarray = None,
):
    """
    Galaxy builder

    This function builds a galaxy object from a star,
    dark matter and gas ParticleSet.

    Parameters
    ----------
    m_s : float
        Star masses
    x_s, y_s, z_s : float
        Star positions.
    vx_s, vy_s, vz_s : float
        Star velocities.
    m_dm : float
        Dark matter masses.
    x_dm, y_dm, z_dm : float
        Dark matter positions.
    vx_dm, vy_dm, vz_dm : float
        Dark matter velocities.
    m_g : float
        Gas masses.
    x_g, y_g, z_g :  float
        Gas positions.
    vx_g, vy_g, vz_g : float
        Gas velocities.
    potential_s : float, default value = None
        Specific potential energy of star particles.
    potential_dm : float, default value = None
        Specific potential energy of dark matter particles.
    potential_g : float, default value = None
        Specific potential energy of gas particles.
    softening_s : float, default value = 0
        Softening radius of stellar particles.
    softening_dm : float, default value = 0
        Softening radius of dark matter particles.
    softening_g : float, default value = 0
        Softening radius of gas particles.

    Return
    ------
    galaxy: object of Galaxy class.
    """
    stars = ParticleSet(
        ParticleSetType.STARS,
        m=m_s,
        x=x_s,
        y=y_s,
        z=z_s,
        vx=vx_s,
        vy=vy_s,
        vz=vz_s,
        softening=softening_s,
        potential=potential_s,
    )
    dark_matter = ParticleSet(
        ParticleSetType.DARK_MATTER,
        m=m_dm,
        x=x_dm,
        y=y_dm,
        z=z_dm,
        vx=vx_dm,
        vy=vy_dm,
        vz=vz_dm,
        softening=softening_dm,
        potential=potential_dm,
    )
    gas = ParticleSet(
        ParticleSetType.GAS,
        m=m_g,
        x=x_g,
        y=y_g,
        z=z_g,
        vx=vx_g,
        vy=vy_g,
        vz=vz_g,
        softening=softening_g,
        potential=potential_g,
    )
    galaxy = Galaxy(stars=stars, dark_matter=dark_matter, gas=gas)
    return galaxy
