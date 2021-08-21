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

from galaxychop import utils

import numpy as np

import pandas as pd

import uttr


# =============================================================================
# COLUMNS CLASS
# =============================================================================
class Columns(enum.Enum):
    """
    Columns name used to decompose galaxies.

    Name and number of the columns that are used to decompose the galaxy
    dynamically.

    Notes
    -----
    The dynamical decomposition is only perform over stellar particles.
    """

    #: mass
    m = 0

    #: y-position
    x = 1
    #: y-position
    y = 2

    #: z-position
    z = 3

    #: x-component of velocity
    vx = 4

    #: y-component of velocity
    vy = 5

    #: z-component of velocity
    vz = 6

    #: softening
    softening = 7

    #: potential energy
    potential = 8

    #: Normalized specific energy of stars
    normalized_energy = 9

    #: Circularity param
    eps = 10

    #: Circularity param r
    eps_r = 11

    @classmethod
    def aslist(cls):
        aslist = list(cls)
        aslist.sort(key=lambda r: r.value)
        return aslist

    @classmethod
    def names(cls):
        return [c.name for c in cls.aslist()]

    @classmethod
    def values(cls):
        return [c.value for c in cls.aslist()]


# =============================================================================
# PARTICLE SET
# =============================================================================


@uttr.s(frozen=True, slots=True, repr=False)
class ParticleSet:

    ptype = uttr.ib(converter=str)

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

    softening: float = attr.ib(converter=float, repr=False)

    has_potential_: bool = attr.ib(init=False)
    kinetic_energy_: np.ndarray = uttr.ib(unit=(u.km / u.s) ** 2, init=False)
    total_energy_: np.ndarray = uttr.ib(unit=(u.km / u.s) ** 2, init=False)
    # UTTRS Orchectration =====================================================

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

    def __attrs_post_init__(self):
        """
        Validate attrs with units.

        Units length validator.

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

    # REDEFINITIONS ===========================================================

    def __repr__(self):
        return (
            f"ParticleSet({self.ptype}, size={len(self)}, "
            f"softening={self.softening}, potentials={self.has_potential_})"
        )

    def __len__(self):
        return len(self.m)

    # UTILITIES ===============================================================

    def to_dataframe(self):
        arr = self.arr_
        data = {
            "ptype": self.ptype,
            "m": arr.m,
            "x": arr.x,
            "y": arr.y,
            "z": arr.z,
            "vx": arr.vx,
            "vy": arr.vy,
            "vz": arr.vz,
            "softening": self.softening,
            "potential": arr.potential if self.has_potential_ else np.nan,
            "kinetic_energy": arr.kinetic_energy_,
            "total_energy": arr.total_energy_
            if self.has_potential_
            else np.nan,
        }
        df = pd.DataFrame(data)
        return df

    def to_numpy(self):
        df = self.to_dataframe()
        del df["ptype"]
        return df.to_numpy()


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

    J_part = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))
    J_star = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))
    Jr_part = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))
    Jr_star = uttr.ib(default=None, unit=(u.kpc * u.km / u.s))

    x = uttr.ib(default=None, unit=u.dimensionless_unscaled)
    y = uttr.ib(default=None, unit=u.dimensionless_unscaled)

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

    # UTILITIES ===============================================================

    def to_dataframe(self):
        return pd.concat(
            [
                self.stars.to_dataframe(),
                self.dark_matter.to_dataframe(),
                self.gas.to_dataframe(),
            ]
        )

    # ENERGY ===============================================================

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
        if not self.has_potential_:
            raise ValueError("Galaxy has not potential energy")
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
        if not self.has_potential_:
            raise ValueError("Galaxy has not potential energy")
        return (
            self.stars.total_energy_,
            self.dark_matter.total_energy_,
            self.gas.total_energy_,
        )

    def potential_energy(self):
        """
        Specific potential energy calculation.

        Calculates the specific potencial energy
        of dark matter, star and gas particles.

        Returns
        -------
        gx : `galaxy object`
            New instanced galaxy specific potencial energy calculated for
            stars, dark matter and gas particles.

        Examples
        --------
        This returns the specific potential energy of stars, dark matter and
        gas particles.

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> gpot = galaxy.potential_energy()
        >>> pot_s, pot_dm, pot_g = gpot.pot_s, gpot.pot_dm, gpot.pot_g

        Note
        ----
        If the potentials are entered when the `galaxy` object is instanced,
        then, the calculation of `potential_energy` will raise a `ValueError`.
        """
        if self.has_potential_:
            raise ValueError("Potentials are already calculated")

        df = self.to_dataframe()
        x = df.x.to_numpy()
        y = df.y.to_numpy()
        z = df.z.to_numpy()
        m = df.m.to_numpy()
        softening = df.softening.max()

        pot = utils.potential(x, y, z, m, softening)

        num_s = len(self.stars)
        num = len(self.stars) + len(self.dark_matter)

        pot_s = pot[:num_s]
        pot_dm = pot[num_s:num]
        pot_g = pot[num:]

        new = as_kwargs(self)

        new.update(
            potential_s=-pot_s * (u.km / u.s) ** 2,
            potential_dm=-pot_dm * (u.km / u.s) ** 2,
            potential_g=-pot_g * (u.km / u.s) ** 2,
        )

        return mkgalaxy(**new)

    def angular_momentum(self, r_cut=None):
        """
        Specific angular momentum.

        Centers the particles with respect to the one with the lower specific
        potential, then, calculates the specific angular momentum of
        dark matter, stars and gas particles.

        Parameters
        ----------
        r_cut : `float`, optional
            The default is ``None``; if provided, it must be
            positive and the rotation matrix `A` is calculated
            from the particles with radii smaller than r_cut.

        Returns
        -------
        gx : `galaxy object`
            New instanced galaxy with all particles centered respect to the
            lowest specific energy one and the addition of J_part, J_star,
            Jr_part and Jr_star.

        Examples
        --------
        This returns the specific potential energy of stars, dark matter and
        gas particles.

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> g_J = galaxy.angular_momentum()
        >>> J_part, J_star= g_J.J_part, g_J.J_star
        >>> Jr_part, Jr_star =  g_J.Jr_part, g_J.Jr_star
        """
        m_s = self.arr_.m_s
        x_s = self.arr_.x_s
        y_s = self.arr_.y_s
        z_s = self.arr_.z_s

        vx_s = self.arr_.vx_s
        vy_s = self.arr_.vy_s
        vz_s = self.arr_.vz_s

        m_dm = self.arr_.m_dm
        x_dm = self.arr_.x_dm
        y_dm = self.arr_.y_dm
        z_dm = self.arr_.z_dm

        vx_dm = self.arr_.vx_dm
        vy_dm = self.arr_.vy_dm
        vz_dm = self.arr_.vz_dm

        m_g = self.arr_.m_g
        x_g = self.arr_.x_g
        y_g = self.arr_.y_g
        z_g = self.arr_.z_g

        vx_g = self.arr_.vx_g
        vy_g = self.arr_.vy_g
        vz_g = self.arr_.vz_g

        pot_s = self.arr_.pot_s
        pot_dm = self.arr_.pot_dm
        pot_g = self.arr_.pot_g

        xs, ys, zs, xdm, ydm, zdm, xg, yg, zg = utils.center(
            m_s,
            x_s,
            y_s,
            z_s,
            m_dm,
            x_dm,
            y_dm,
            z_dm,
            m_g,
            x_g,
            y_g,
            z_g,
            pot_s,
            pot_dm,
            pot_g,
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

        J_star = np.array(
            [
                pos_rot_s_y * vel_rot_s_z - pos_rot_s_z * vel_rot_s_y,
                pos_rot_s_z * vel_rot_s_x - pos_rot_s_x * vel_rot_s_z,
                pos_rot_s_x * vel_rot_s_y - pos_rot_s_y * vel_rot_s_x,
            ]
        )

        J_dark = np.array(
            [
                pos_rot_dm_y * vel_rot_dm_z - pos_rot_dm_z * vel_rot_dm_y,
                pos_rot_dm_z * vel_rot_dm_x - pos_rot_dm_x * vel_rot_dm_z,
                pos_rot_dm_x * vel_rot_dm_y - pos_rot_dm_y * vel_rot_dm_x,
            ]
        )

        J_gas = np.array(
            [
                pos_rot_g_y * vel_rot_g_z - pos_rot_g_z * vel_rot_g_y,
                pos_rot_g_z * vel_rot_g_x - pos_rot_g_x * vel_rot_g_z,
                pos_rot_g_x * vel_rot_g_y - pos_rot_g_y * vel_rot_g_x,
            ]
        )

        J_part = np.concatenate([J_star, J_dark, J_gas], axis=1)

        Jr_star = np.sqrt(J_star[0, :] ** 2 + J_star[1, :] ** 2)

        Jr_part = np.sqrt(J_part[0, :] ** 2 + J_part[1, :] ** 2)

        new = attr.asdict(self, recurse=False)
        del new["arr_"]
        new.update(
            J_part=J_part * u.kpc * u.km / u.s,
            J_star=J_star * u.kpc * u.km / u.s,
            Jr_part=Jr_part * u.kpc * u.km / u.s,
            Jr_star=Jr_star * u.kpc * u.km / u.s,
        )

        return Galaxy(**new)

    def jcirc(self, bin0=0.05, bin1=0.005):
        """
        Circular angular momentum.

        Calculation of the points to build the function of the circular
        angular momentum.

        Parameters
        ----------
        bin0 : `float`. Default=0.05
            Size of the specific energy bin of the inner part of the galaxy,
            in the range of (-1, -0.1) of the normalized energy.
        bin1 : `float`. Default=0.005
            Size of the specific energy bin of the outer part of the galaxy,
            in the range of (-0.1, 0) of the normalized energy.

        Returns
        -------
        gx : `galaxy object`
            New instanced galaxy with `x`, being the normalized specific
            energy for the particle with the maximum z-specific angular
            momentum component per the bin, and `y` beign the maximum of
            z-specific angular momentum component.
            See section Notes for more details.

        Notes
        -----
            The `x` and `y` are calculated from the binning in the
            normalized specific energy. In each bin, the particle with the
            maximum value of z-component of standardized specific angular
            momentum is selected. This value is assigned to the `y` parameter
            and its corresponding normalized specific energy pair value to
            `x`.

        Examples
        --------
        This returns the normalized specific energy for the particle with
        the maximum z-component of the normalized specific angular momentum
        per bin (`x`) and the maximum value of the z-component of the
        normalized specific angular momentum per bin (`y`)

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> g_Jcirc = galaxy.jcirc()
        >>> x, y = g_Jcirc.x, g_Jcirc.y

        """
        Etot_s = self.energy[0].value
        Etot_dm = self.energy[1].value
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
        new.update(x=u.Quantity(x), y=u.Quantity(y))

        return Galaxy(**new)

    @property
    def paramcirc(self):
        """
        Circularity parameter calculation.

        Return
        ------
        tuple : `float`
            (E_star, eps, eps_r): Normalized specific energy of the stars,
            circularity parameter (J_z/J_circ), J_p/J_circ.
            Shape(n_s, 1). Unit: dimensionless

        Notes
        -----
        J_z : z-component of normalized specific angular momentum.

        J_circ : Specific circular angular momentum.

        J_p : Projection on the xy plane of the normalized specific angular
        momentum.

        Examples
        --------
        This returns the normalized specific energy of stars (E_star), the
        circularity parameter (eps : J_z/J_circ) and
        eps_r: (J_p/J_circ).

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> E_star, eps, eps_r = galaxy.paramcirc

        """
        Etot_s = self.energy[0].value
        Etot_dm = self.energy[1].value
        Etot_g = self.energy[2].value

        E_tot = np.hstack([Etot_s, Etot_dm, Etot_g])

        E_star_ = np.full(len(Etot_s), np.nan)
        eps_ = np.full(len(Etot_s), np.nan)
        eps_r_ = np.full(len(Etot_s), np.nan)

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
        down3 = np.max(np.abs(ang_momentum.Jr_part[neg][fin]))
        Jr_star_norm = up3 / down3

        # We do the interpolation to calculate the J_circ.
        # spl = InterpolatedUnivariateSpline(
        #    self.jcirc().arr_.x,
        #    self.jcirc().arr_.y,
        #    k=1,
        # )

        # Calculates of the circularity parameter Lz/Lc.
        # eps = J_star_ / spl(E_star)
        jcir = self.jcirc().arr_
        eps = Jz_star_norm / np.interp(E_star, jcir.x, jcir.y)

        # Calculates the same for Lp/Lc.
        # eps_r = Jr_star_ / spl(E_star)
        eps_r = Jr_star_norm / np.interp(E_star, jcir.x, jcir.y)

        # We remove particles that have circularity < -1 and circularity > 1.
        (mask,) = np.where((eps <= 1.0) & (eps >= -1.0))

        E_star_[neg_star[fin_star[mask]]] = E_star[mask]
        eps_[neg_star[fin_star[mask]]] = eps[mask]
        eps_r_[neg_star[fin_star[mask]]] = eps_r[mask]

        return (E_star_, eps_, eps_r_)


# =============================================================================
# API FUNCTIONS
# =============================================================================


def as_kwargs(galaxy):
    def _filter_internals(attribute, value):
        return attribute.init

    def _pset_as_kwargs(pset, suffix):
        return {f"{k}_{suffix}": v for k, v in pset.items() if k != "ptype"}

    gkwargs = attr.asdict(galaxy, recurse=True, filter=_filter_internals)

    stars_kws = _pset_as_kwargs(gkwargs.pop("stars"), "s")
    dark_matter_kws = _pset_as_kwargs(gkwargs.pop("dark_matter"), "dm")
    gas_kws = _pset_as_kwargs(gkwargs.pop("gas"), "g")

    gkwargs.update(**stars_kws, **dark_matter_kws, **gas_kws)

    del (
        gkwargs["J_part"],
        gkwargs["J_star"],
        gkwargs["Jr_part"],
        gkwargs["Jr_star"],
        gkwargs["x"],
        gkwargs["y"],
    )

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
        "stars",
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
        "dark_matter",
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
        "gas",
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
