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

from . import plot


# =============================================================================
# PARTICLE SET
# =============================================================================
class ParticleSetType(enum.Enum):

    STARS = 0
    DARK_MATTER = 1
    GAS = 2


@uttr.s(frozen=True, slots=True, repr=False)
class ParticleSet:

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
        particle sets length validator.

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

    def to_dataframe(self, columns=None):
        arr = self.arr_
        mkcolumns = {
            "ptype": lambda: self.ptype.name,
            "ptypev": lambda: self.ptype.value,
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
        columns = mkcolumns.keys() if columns is None else columns
        data = {}
        for colname in columns:
            column_make = mkcolumns[colname]
            data[colname] = column_make()
        return pd.DataFrame(data)


# =============================================================================
# GALAXY CLASS
# =============================================================================


@uttr.s(frozen=True)
class Galaxy:
    """
    Galaxy class.

    Builds a galaxy object from masses, positions, and
    velocities of particles (stars, dark matter and gas).

    Parameters
    ----------

    J_part : `Quantity`
        Total specific angular momentum of all particles (stars, dark matter
        and gas).
        Shape: (n,3). Default units: kpc*km/s
    J_star : `Quantity`
        Total specific angular momentum of stars.
        Shape: (n_s,1). Default unit: kpc*km/s
    Jr_part : `Quantity`
        Projection of the total specific angular momentum in the xy plane for
        all particles.
        Shape: (n,1). Default unit: kpc*km/s
    Jr_star : `Quantity`
        Projection of the specific angular momentum of stars in the xy plane.
        Shape: (n_s,1). Default unit: kpc*km/s
    x : `Quantity`
        Normalized specific energy for the particle with the maximum
        z-component of the normalized specific angular momentum per bin.
        Default unit: dimensionless
    y : `Quantity`
        Maximum value of the z-component of the normalized specific angular
        momentum per bin.
        Default units: dimensionless

    Attributes
    ----------
    arr_: `uttr.ArrayAccessor`
        Original array accessor object create by the *uttr* library.
        Array accesor: it converts uttr attributes to the default unit and
        afterward to a `numpy.ndarray`.
        For more information see: https://pypi.org/project/uttrs/
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

    # UTILITIES ===============================================================

    def to_dataframe(self, *, ptypes=None, columns=None):
        mkptypes = {
            "stars": self.stars.to_dataframe,
            "dark_matter": self.dark_matter.to_dataframe,
            "gas": self.gas.to_dataframe,
        }

        ptypes = mkptypes.keys() if ptypes is None else ptypes

        parts = []
        for ptype in ptypes:
            maker = mkptypes[ptype]
            df = maker(columns=columns)
            parts.append(df)

        return pd.concat(parts)

    @property
    def plot(self):
        """Plot accessor."""
        return plot.GalaxyPlotter(self)

    # ENERGY ===============================================================

    @property
    def is_aligned(self):
        return util.is_star_aligned(self)

    @property
    def kinetic_energy_(self):
        """Specific kinetic energy
        of stars, dark matter and gas particles.

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
    def Jstar_(self):
        return self.stars.angular_momentum_

    @property
    def Jdark_matter_(self):
        return self.dark_matter.angular_momentum_

    @property
    def Jgas_(self):
        return self.gas.angular_momentum_

    def jcirc(self, bin0=0.05, bin1=0.005):
        """
        Processing energy and angular momentum.

        Calculation of Normalized specific energy of the stars,
        circularity parameter calculation, projected circularity parameter,
        and the points to build the function of the circular angular momentum.

        Parameters
        ----------
        bin0 : `float`. Default=0.05
            Size of the specific energy bin of the inner part of the galaxy,
            in the range of (-1, -0.1) of the normalized energy.
        bin1 : `float`. Default=0.005
            Size of the specific energy bin of the outer part of the galaxy,
            in the range of (-0.1, 0) of the normalized energy.

        Return
        ------
        tuple : `float`
            (E_star_norm, eps, eps_r, x, y): Normalized specific energy of the stars,
            circularity parameter (J_z/J_circ), projected circularity parameter
            (J_p/J_circ), the normalized specific energy for the particle with
            the maximum z-specific angular momentum component per the bin (x),
            and the maximum of z-specific angular momentum component (y).
            See section Notes for more details.
            Shape(n_s, 1). Unit: dimensionless

        Notes
        -----
        The `x` and `y` are calculated from the binning in the
        normalized specific energy. In each bin, the particle with the
        maximum value of z-component of standardized specific angular
        momentum is selected. This value is assigned to the `y` parameter
        and its corresponding normalized specific energy pair value to `x`.

        Examples
        --------
        This returns the normalized specific energy of stars (E_star_norm), the
        circularity parameters (eps : J_z/J_circ and
        eps_r: J_p/J_circ), and the normalized specific energy for the particle with
        the maximum z-component of the normalized specific angular momentum
        per bin (`x`) and the maximum value of the z-component of the
        normalized specific angular momentum per bin (`y`).

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> E_star_norm, eps, eps_r, x, y = galaxy.jcir(bin0=0.05, bin1=0.005)
        """

        df = self.to_dataframe(["ptypev", "total_energy", "Jx", "Jy", "Jz"])
        Jr_part = np.sqrt(df.Jx **2 + df.Jy **2)
        E_tot = df.total_energy

        # Remove the particles that are not bound: E > 0 and with E = -inf.
        (bound,) = np.where((E_tot <= 0.0) & (E_tot != -np.inf))

        # Normalize the two variables: E between 0 and 1; Jz between -1 and 1.
        E = E_tot[bound] / np.abs(np.min(E_tot[bound]))
        Jz = df.Jz[bound] / np.max(np.abs(df.Jz[bound]))

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

        #Stars particles
        df_star = df[df.ptypev == ParticleSetType.STARS.value]
        Jr_star = np.sqrt(df_star.Jx **2 + df_star.Jy **2)
        Etot_s = df_star.total_energy

        # Remove the star particles that are not bound: E > 0 and with E = -inf.
        (bound_star,) = np.where((Etot_s <= 0.0) & (Etot_s != -np.inf))

        # Normalize E, Jz and Jr for the stars.
        E_star_norm = Etot_s[bound_star] / np.abs(np.min(E_tot[bound]))
        Jz_star_norm = df_star.Jz[bound_star] / np.max(np.abs(df.Jz[bound]))
        Jr_star_norm = Jr_star[bound_star] / np.max(np.abs(Jr_part[bound]))

        # Calculates of the circularity parameters Jz/Jcirc and Jproy/Jcirc.
        j_circ = np.interp(E_star_norm, x, y)
        eps = Jz_star_norm / j_circ
        eps_r = Jr_star_norm / j_circ

        # We remove particles that have circularity < -1 and circularity > 1.
        (mask,) = np.where((eps <= 1.0) & (eps >= -1.0))

        E_star_norm_ = np.full(len(Etot_s), np.nan)
        eps_ = np.full(len(Etot_s), np.nan)
        eps_r_ = np.full(len(Etot_s), np.nan)

        E_star_norm_[bound_star[mask]] = E_star_norm[mask]
        eps_[bound_star[mask]] = eps[mask]
        eps_r_[bound_star[mask]] = eps_r[mask]

        return (E_star_norm_, eps_, eps_r_, x, y)


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
