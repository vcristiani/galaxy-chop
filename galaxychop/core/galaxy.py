# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Module galaxy-chop."""

# =============================================================================
# IMPORTS
# =============================================================================

import enum
from collections import OrderedDict, defaultdict

from astropy import units as u
from astropy.table import Table

import attr

import numpy as np

import pandas as pd

import uttr

from .. import constants as const

# =============================================================================
# EXCEPTIONS
# =============================================================================


class NoGravitationalPotentialError(ValueError):
    """Used to inform that a calculation cannot be performed if the galaxy \
    does not have potential."""


# =============================================================================
# PARTICLE SET TYPE
# =============================================================================


class ParticleSetType(enum.IntEnum):
    """
    Name of the particle type.

    Name and number that are used to describe the particle
    type in the ``ParticleSet class``.

    """

    STARS = 0
    DARK_MATTER = 1
    GAS = 2

    @classmethod
    def mktype(cls, v):
        """Create a ParticleSetType from name, or value."""
        if isinstance(v, ParticleSetType):
            return v
        if isinstance(v, str):
            v = v.upper()
        for p in ParticleSetType:
            if v in (p.name, p.value):
                return p
        raise ValueError(f"Can't coerce {v} into ParticleSetType ")

    def humanize(self):
        """Particle type name in lower case."""
        return self.name.lower()


# =============================================================================
# PARTICLE SET
# =============================================================================


@uttr.s(frozen=True, slots=True, repr=False)
class ParticleSet:
    """
    ParticleSet class.

    Creates a set particles of a particular type (stars, dark matter or gas)
    using masses, positions, velocities and potential energy.

    Parameters
    ----------
    ptype : ParticleSetType
        Indicates if this set corresponds to stars, dark matter or gas.
    m : Quantity
        Particle masses. Shape: (n,1). Default unit: M_sun
    x, y, z : Quantity
        Positions. Shapes: (n,1). Default unit: kpc.
    vx, vy, vz : Quantity
        Velocities. Shapes: (n,1). Default unit: km/s.
    potential : Quantity, default value = 0
        Specific potential energy of particles. Shape: (n,1). Default unit:
        (km/s)**2.
    softening : Quantity. Default value = 0
        Softening radius of particles. Shape: (1,). Default unit: kpc.
    kinetic_energy : Quantity
        Specific kinetic energy of particles. Shape: (n,1). Default unit:
        (km/s)**2.
    total_energy : Quantity
        Specific total energy of particles. Shape: (n,1). Default unit:
        (km/s)**2.
    Jx_, Jy_, Jz_ : Quantity
        Components of angular momentum of particles. Shapes: (n,1). Default
        units: kpc*km/s.
    has_potential_ : bool.
        Indicates if the specific potential energy is computed.
    arr_ : Instances of ``ArrayAccessor``
        Access to the attributes (defined with uttrs) of the provided instance,
        and if they are of astropy.units.Quantity type it converts them into
        numpy.ndarray.

    """

    ptype = uttr.ib(validator=attr.validators.instance_of(ParticleSetType))

    m: np.ndarray = uttr.ib(unit=u.Msun, converter=np.copy)
    x: np.ndarray = uttr.ib(unit=u.kpc, converter=np.copy)
    y: np.ndarray = uttr.ib(unit=u.kpc, converter=np.copy)
    z: np.ndarray = uttr.ib(unit=u.kpc, converter=np.copy)
    vx: np.ndarray = uttr.ib(unit=(u.km / u.s), converter=np.copy)
    vy: np.ndarray = uttr.ib(unit=(u.km / u.s), converter=np.copy)
    vz: np.ndarray = uttr.ib(unit=(u.km / u.s), converter=np.copy)

    potential: np.ndarray = uttr.ib(
        unit=(u.km / u.s) ** 2,
        validator=attr.validators.optional(
            attr.validators.instance_of(np.ndarray)
        ),
        converter=(lambda v: np.copy(v) if v is not None else v),
        repr=False,
    )

    softening: float = uttr.ib(unit=u.kpc, converter=float, repr=False)

    has_potential_: bool = uttr.ib(init=False)

    kinetic_energy_: np.ndarray = uttr.ib(unit=(u.km / u.s) ** 2, init=False)
    total_energy_: np.ndarray = uttr.ib(unit=(u.km / u.s) ** 2, init=False)

    # angular momentum
    Jx_: np.ndarray = uttr.ib(unit=(u.kpc * u.km / u.s), init=False)
    Jy_: np.ndarray = uttr.ib(unit=(u.kpc * u.km / u.s), init=False)
    Jz_: np.ndarray = uttr.ib(unit=(u.kpc * u.km / u.s), init=False)

    # INITIALIZATION ==========================================================

    @has_potential_.default
    def _has_potential__default(self):
        return self.potential is not None

    @kinetic_energy_.default
    def _kinetic_energy__default(self):
        arr = self.arr_
        ke = 0.5 * (arr.vx**2 + arr.vy**2 + arr.vz**2)
        return ke

    @total_energy_.default
    def _total_energy__default(self):
        if not self.has_potential_:
            return

        arr = self.arr_
        kenergy = arr.kinetic_energy_
        penergy = arr.potential

        return kenergy + penergy

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

        This method determines that the lengths of the different attributes of
        particles that are of the same family are the same.
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

        for field in attr.astuple(self):
            if isinstance(field, np.ndarray):
                field.setflags(write=False)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        cls_name = type(self).__name__
        return (
            f"<{cls_name} {self.ptype.name!r}, size={len(self)}, "
            f"softening={self.softening.value}, "
            f"potentials={self.has_potential_}>"
        )

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return len(self.m)

    # PROPERTIES ==============================================================

    @property
    def total_mass(self):
        """Total mass of all particles in the set in units of M_sun."""
        return np.sum(self.m)

    @property
    def angular_momentum_(self):
        """Components of specific angular momentum in units of kpc*km/s."""
        arr = self.arr_
        return np.array([arr.Jx_, arr.Jy_, arr.Jz_]) * (u.kpc * u.km / u.s)

    # PUBLIC METHODS ==========================================================

    def get_value_makers(self):
        """
        Build a dictionary of value generator functions for particle attribute.

        This method constructs a mapping where each key corresponds to a
        particle property name (e.g., position, velocity, energy), and the
        associated value is a callable that returns a NumPy array with the
        corresponding values for all particles. Arrays are returned directly
        from the internal storage or created on demand (e.g., filling with NaNs
        if a quantity is unavailable).

        Returns
        -------
        dict of {str: callable}
            Dictionary mapping attribute names to zero-argument functions that
            return NumPy arrays of values for all particles. Includes:

            - ``ptype`` : human-readable particle type.
            - ``ptypev`` : numeric particle type value.
            - ``m`` : particle masses.
            - ``x``, ``y``, ``z`` : spatial coordinates.
            - ``vx``, ``vy``, ``vz`` : velocity components.
            - ``softening`` : softening length.
            - ``potential`` : gravitational potential (NaN if unavailable).
            - ``kinetic_energy`` : kinetic energy per particle.
            - ``total_energy``: total energy per particle (NaN if unavailable).
            - ``Jx``, ``Jy``, ``Jz`` : angular momentum components.
        """
        arr = self.arr_
        value_makers = {
            "ptype": lambda: np.full(len(self), self.ptype.humanize()),
            "ptypev": lambda: np.full(len(self), self.ptype.value),
            "m": lambda: arr.m,
            "x": lambda: arr.x,
            "y": lambda: arr.y,
            "z": lambda: arr.z,
            "vx": lambda: arr.vx,
            "vy": lambda: arr.vy,
            "vz": lambda: arr.vz,
            "softening": lambda: arr.softening,
            "potential": lambda: (
                arr.potential
                if self.has_potential_
                else np.full(len(self), np.nan)
            ),
            "kinetic_energy": lambda: arr.kinetic_energy_,
            "total_energy": lambda: (
                arr.total_energy_
                if self.has_potential_
                else np.full(len(self), np.nan)
            ),
            "Jx": lambda: arr.Jx_,
            "Jy": lambda: arr.Jy_,
            "Jz": lambda: arr.Jz_,
        }
        return value_makers

    def to_dict(self, *, attributes=None):
        """
        Convert the galaxy into a dict of array with coerced units.

        Parameters
        ----------
        attributes: tuple, default value = None
            Dictionary keys of ParticleSet parameters used to create the dict
            If it's None, the data frame is constructed from all the
            parameters of the ``ParticleSet class``.

        Return
        ------
        dict :
            dictionary with coerced units.

        """

        def _make_elements(basename, value):
            dims = np.ndim(value)
            elems = OrderedDict()
            if dims == 1:
                elems[basename] = value
            elif dims == 2:
                for idx, column in enumerate(value.T):
                    name = f"{basename}_{idx}"
                    elems[name] = column

            return elems

        value_makers = self.get_value_makers()
        attributes = value_makers.keys() if attributes is None else attributes
        the_dict = OrderedDict()
        for aname in attributes:
            mkvalue = value_makers[aname]
            avalue = mkvalue()
            avalue.setflags(write=True)
            avalue_dict = _make_elements(basename=aname, value=avalue)
            the_dict.update(avalue_dict)
        return the_dict

    def to_dataframe(self, *, attributes=None):
        """
        Convert to pandas data frame.

        This method constructs a data frame with the particles and parameters
        of ``ParticleSet class``.

        Parameters
        ----------
        attributes: tuple, default value = None
            Dictionary keys of ParticleSet parameters used to create the data
            frame. If it's None, the data frame is constructed from all the
            parameters of the ``ParticleSet class``.

        Return
        ------
        DataFrame : pandas data frame
            Data frame of the particles with the selected parameters.

        """
        the_dict = self.to_dict(attributes=attributes)
        return pd.DataFrame(the_dict)

    # CONVERSION METHODS ======================================================

    def copy(self):
        """Make a copy of the ParticleSet."""
        cls = type(self)
        new = cls(
            ptype=self.ptype,
            m=self.m,
            x=self.x,
            y=self.y,
            z=self.z,
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
            potential=self.potential,
            softening=float(self.softening.value),
        )
        return new

    # PRIVATE/SPECIAL METHODS =================================================

    def _gchop_h5_(self):
        """
        Extract HDF5 serialization data for this ParticleSet.

        Returns metadata about the particle set type and a DataFrame
        containing the particle data that should be persisted.

        Returns
        -------
        metadata : dict
            Dictionary containing particle set metadata:
            - 'pset_type': class name of the particle set
            - 'ptype': particle type (stars, dark_matter, gas)
            - 'has_potential': whether potential is computed
        dataframe : pd.DataFrame
            DataFrame with particle attributes to be saved

        """
        # Get attributes to serialize (exclude softening as it's set at read time)
        cls = type(self)
        attributes = [
            f.name
            for f in attr.fields(cls)
            if f.init and f.name not in ("ptype", "softening")
        ]

        if not self.has_potential_:
            attributes.remove("potential")

        # Create metadata
        metadata = {
            "pset_type": cls.__name__,
            "ptype": self.ptype.name,
            "has_potential": bool(self.has_potential_),
        }

        # Create DataFrame and AstropyTable
        df = self.to_dataframe(attributes=attributes)
        df.insert(0, "id", df.index.to_numpy())
        table = Table.from_pandas(df)

        del df

        return metadata, table


# =============================================================================
# GALAXY CLASS
# =============================================================================


@uttr.s(frozen=True, repr=False)
class Galaxy:
    """
    Galaxy class.

    Builds a galaxy object from a ``ParticleSet`` for each type
    of particle (stars, dark matter and gas).

    Parameters
    ----------
    stars : ``ParticleSet``
        Instance of ``ParticleSet`` with stars particles.
    dark_matter : ``ParticleSet``
        Instance of ``ParticleSet`` with dark matter particles.
    gas : ``ParticleSet``
        Instance of ParticleSet with gas particles.

    Attributes
    ----------
    has_potential_: bool
        Indicates if this Galaxy instance has the potential energy computed.
    is_aligned_ : bool.
        Indicates if this Galaxy instance has been already aligned i.e. the
        Z-axis is parallel to the minor axis of the particle system.
    is_centered_ : bool.
        Indicates if this Galaxy instance has been already centered i.e.
        the most bound particle defines the origin of the system.

    """

    PSET_CLS = ParticleSet

    stars = uttr.ib(validator=attr.validators.instance_of(ParticleSet))
    dark_matter = uttr.ib(validator=attr.validators.instance_of(ParticleSet))
    gas = uttr.ib(validator=attr.validators.instance_of(ParticleSet))

    has_potential_ = attr.ib(init=False)

    # INITIALIZATION ==========================================================

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

            if not isinstance(pset, self.PSET_CLS):
                raise TypeError(
                    f"particle set {pset!r} "
                    f"must be of type {self.PSET_CLS.__name__}"
                )

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return len(self.stars) + len(self.dark_matter) + len(self.gas)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        cls_name = type(self).__name__
        stars_repr = f"stars={len(self.stars)}"
        dm_repr = f"dark_matter={len(self.dark_matter)}"
        gas_repr = f"gas={len(self.gas)}"
        has_pot = f"potential={self.has_potential_}"
        return (
            f"<{cls_name} {stars_repr}, {dm_repr}, "
            f"{gas_repr}, {has_pot}>"
        )

    # PROPERTIES ==============================================================

    @property
    def total_mass(self):
        """
        Total mass for each particle type as a DataFrame.

        Returns
        -------
        DataFrame : pandas DataFrame
            DataFrame with particle types as index and their total masses
            in M_sun units.

        Examples
        --------
        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> galaxy.total_mass
                      total_mass
        stars              1.2e10
        dark_matter        5.3e11
        gas                2.1e09
        """
        data = {
            "total_mass": [
                self.stars.total_mass.value,
                self.dark_matter.total_mass.value,
                self.gas.total_mass.value,
            ]
        }
        index = [
            self.stars.ptype.humanize(),
            self.dark_matter.ptype.humanize(),
            self.gas.ptype.humanize(),
        ]
        return pd.DataFrame(data, index=index)

    @property
    def plot(self):
        """Plot accessor."""
        if not hasattr(self, "_plot"):
            from . import plot  # noqa

            plotter = plot.GalaxyPlotter(self)
            super().__setattr__("_plot", plotter)
        return self._plot

    @property
    def kinetic_energy_(self):
        """
        Specific kinetic energy of stars, dark matter and gas particles.

        Returns
        -------
        tuple : Quantity
            (k_s, k_dm, k_g): Specific kinetic energy of stars, dark matter and
            gas respectively. Shape(n_s, n_dm, n_g). Unit: (km/s)**2

        Examples
        --------
        This returns the specific kinetic energy of stars, dark matter and gas
        particles respectively.

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> k_s, k_dm, k_g = galaxy.kinetic_energy_
        """
        return (
            self.stars.kinetic_energy_,
            self.dark_matter.kinetic_energy_,
            self.gas.kinetic_energy_,
        )

    @property
    def potential_energy_(self):
        """
        Specific potential energy of stars, dark matter and gas particles.

        This property doesn't compute the potential energy, only returns its
        value if it is already computed, i.e. ``has_potential_`` is True. To
        compute the potential use the
        ``galaxychop.preproc.potential_energy`` module.

        Returns
        -------
        tuple : Quantity
            (p_s, p_dm, p_g): Specific potential energy of stars, dark matter
            and gas respectively. Shape(n_s, n_dm, n_g). Unit: (km/s)**2

        Examples
        --------
        This returns the specific potential energy of stars, dark matter and
        gas particles respectively.

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> pot = gchop.preproc.potential_energy.Potentializer(
            backend="frotran")
        >>> galaxy_with_potential = pot.transform(galaxy)
        >>> p_s, p_dm, p_g = galaxy_with_potential.potential_energy_
        """
        if self.has_potential_:
            return (
                self.stars.potential,
                self.dark_matter.potential,
                self.gas.potential,
            )
        else:
            raise NoGravitationalPotentialError(
                "Galaxy does not have the potential energy calculated"
            )

    @property
    def total_energy_(self):
        """
        Specific total energy calculation.

        Calculates the specific total energy of dark matter, star and gas
        particles.

        Returns
        -------
        tuple : Quantity
            (Etot_s, Etot_dm, Etot_g): Specific total energy of stars, dark
            matter and gas respectively. Shape(n_s, n_dm, n_g). Unit: (km/s)**2

        Examples
        --------
        This returns the specific total energy of stars, dark matter and gas
        particles respectively.

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> E_s, E_dm, E_g = galaxy.total_energy_

        """
        if self.has_potential_:
            return (
                self.stars.total_energy_,
                self.dark_matter.total_energy_,
                self.gas.total_energy_,
            )
        else:
            raise NoGravitationalPotentialError(
                "Galaxy does not have the potential energy calculated"
            )

    @property
    def angular_momentum_(self):
        """
        Specific angular momentum calculation.

        Compute the specific angular momentum of stars, dark matter and gas
        particles.

        Returns
        -------
        tuple : `Quantity`
            (J_s, J_dm, J_g): Specific angular momentum of stars, dark
            matter and gas respectively. Shape(n_s, n_dm, n_g).
            Unit: (kpc * km / s)

        Examples
        --------
        This returns the specific angular momentum of stars, dark matter and
        gas particles respectively.

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> J_s, J_dm, J_g = galaxy.angular_momentum_

        """
        return (
            self.stars.angular_momentum_,
            self.dark_matter.angular_momentum_,
            self.gas.angular_momentum_,
        )

    # PUBLIC METHODS ==========================================================

    def to_dataframe(self, *, ptypes=None, attributes=None, sdynamics=True):
        """
        Convert the galaxy to pandas DataFrame.

        This method builds a data frame from the particles of the Galaxy.

        Parameters
        ----------
        ptypes: tuple, default value = None
            Strings indicating the ParticleSetType to include. If it's None,
            all particle types are included.
        attributes: tuple, default value = None
            Dictionary keys of ParticleSet parameters used to create the data
            frame. If it's None, the data frame is constructed from all the
            parameters of the ``ParticleSet class``.

        Return
        ------
        DataFrame : pandas data frame
            Data frame of Galaxy with selected particles and parameters.

        """
        psets = [self.stars, self.dark_matter, self.gas]

        parts = []
        for pset in psets:
            if ptypes is None or pset.ptype.humanize() in ptypes:
                df = pset.to_dataframe(attributes=attributes)
                parts.append(df)

        return pd.concat(parts, ignore_index=True)

    def to_hdf5(self, path_or_stream, *, metadata=None, **kwargs):
        """
        Shortcut to ``galaxychop.io.to_hdf5()``.

        It is responsible for storing a galaxy in HDF5 format. The procedure
        only stores the attributes ``m``, ``x``, ``y``, ``z``, ``vx``, ``vy``
        and ``vz``,  since all the other attributes can be derived from these,
        and the ``softenings`` can be arbitrarily changed at the galaxy
        creation/reading process

        Parameters
        ----------
        path_or_stream : str or file like.
            Path or file like objet to the h5 to store the galaxy.
        metadata : dict or None (default None)
            Extra metadata to store in the h5 file.
        kwargs :
            Extra arguments to the function
            ``astropy.io.misc.hdf5.write_table_hdf5()``

        """
        from .. import io

        return io.to_hdf5(
            path_or_stream=path_or_stream,
            galaxy=self,
            metadata=metadata,
            **kwargs,
        )

    def to_dict(self, *, ptypes=None, attributes=None):
        """
        Convert the galaxy to dict with information as a numpy array with \
        coerced units.

        Parameters
        ----------
        ptypes: tuple, default value = None
            Strings indicating the ParticleSetType to include. If it's None,
            all particle types are included.
        attributes: tuple, default value = None
            Dictionary keys of ParticleSet parameters used to create the dict.
            If it's None, the data frame is constructed from all the
            parameters of the ``ParticleSet class``.

        Return
        ------
        dict : pandas data frame
            dictionary with all data of stars, dark_matter and gas as numpy
            array with coerced units

        """
        psets = [self.stars, self.dark_matter, self.gas]

        the_dict = {}
        for pset in psets:
            ptype = pset.ptype.humanize()
            if ptypes is None or ptype in ptypes:
                p_dict = pset.to_dict(attributes=attributes)
                the_dict[ptype] = p_dict

        return the_dict

    def disassemble(self):
        """
        Convert all the attributes of the galaxy into a play dict.

        The resulting dict can be used to build a new galaxy with
        ``galaxychop.mkgalaxy``.

        Parameters
        ----------
        galaxy: Galaxy
            Instance of Galaxy.

        Returns
        -------
        dict
            Dictionary with ``galaxy`` attributes.

        """
        attributes = {
            fname: fvalue
            for fname, fvalue in attr.fields_dict(ParticleSet).items()
            if fvalue.init and fname != "ptype"
        }

        def _flat(pset_dict, suffix):
            the_flatten_dict = {}
            for k, v in pset_dict.items():
                flat_k = f"{k}_{suffix}"
                flat_v = v[0] if np.isscalar(attributes[k].converter) else v
                the_flatten_dict[flat_k] = flat_v
            return the_flatten_dict

        the_dict = self.to_dict(attributes=attributes.keys())
        stars_kws = _flat(the_dict["stars"], "s")
        dark_matter_kws = _flat(the_dict["dark_matter"], "dm")
        gas_kws = _flat(the_dict["gas"], "g")

        disassembled = dict(**stars_kws, **dark_matter_kws, **gas_kws)
        return disassembled

    def stellar_dynamics(
        self,
        *,
        bin0=const.SD_DEFAULT_CBIN[0],
        bin1=const.SD_DEFAULT_CBIN[1],
        reassign=const.SD_DEFAULT_REASSIGN,
        runtime_warnings=const.SD_RUNTIME_WARNING_ACTION,
    ):
        """
        Calculate galaxy stars particles circularity information.

        Shortcut to ``galaxychop.core.sdynamics.stellar_dynamics()``.

        Calculation of Normalized specific energy of the stars, z-component
        normalized specific angular momentum of the stars, circularity
        parameter, projected circularity parameter, and the points to build the
        function of the circular angular momentum.

        Parameters
        ----------
        bin0 : float. Default=0.05
            Size of the specific energy bin of the inner part of the galaxy,
            in the range of (-1, -0.1) of the normalized energy.
        bin1 : float. Default=0.005
            Size of the specific energy bin of the outer part of the galaxy,
            in the range of (-0.1, 0) of the normalized energy.
        reassign : list. Default=False
            It allows to define what to do with stellar particles with
            circularity parameter values >1 or <-1. True reassigns the value
            to 1 or -1, depending on the case. False discards these particles.
        runtime_warnings : Any warning filter action (default "ignore")
            stellar_synamics usually launches RuntimeWarning during the eps
            (J_z/J_circ) calculation because there may be some particle with
            jcirc=0. By default the function decides to ignore these warnings.
            `runtime_warnings` can be set to any valid "action" in the python
            warnings module.

        Return
        ------
        GalaxyStellarDynamics :
            Circularity attributes of the star components of the galaxy

        Notes
        -----
        The `x` and `y` are calculated from the binning in the normalized
        specific energy. In each bin, the particle with the maximum value of
        z-component of standardized specific angular momentum is selected.
        This value is assigned to the `y` parameter and its corresponding
        normalized specific energy pair value to `x`.

        Examples
        --------
        This returns the normalized specific energy of stars (E_star_norm), the
        z-component normalized specific angular momentum of the stars
        (Jz_star_norm), the circularity parameters (eps : J_z/J_circ and
        eps_r: J_p/J_circ), and the normalized specific energy for the particle
        with the maximum z-component of the normalized specific angular
        momentum per bin (`x`) and the maximum value of the z-component of the
        normalized specific angular momentum per bin (`y`).

        >>> import galaxychop as gchop
        >>> galaxy = gchop.Galaxy(...)
        >>> gsd = galaxy.stellar_dynamics(
        ...     bin0=0.05, bin1=0.005, reassign=False
        ... )
        >>> gsd
        GalaxyStellarDynamics(
            normalized_star_energy=[...],
            normalized_star_Jz=[...],
            eps=[...],
            eps_r=[...]
        )

        """
        from . import sdynamics

        return sdynamics.stellar_dynamics(
            self,
            bin0=bin0,
            bin1=bin1,
            reassign=reassign,
            runtime_warnings=runtime_warnings,
        )

    # CONVERSION METHODS ======================================================

    def copy(self):
        """Make a copy of the Galaxy."""
        cls = type(self)
        new = cls(
            stars=self.stars.copy(),
            dark_matter=self.dark_matter.copy(),
            gas=self.gas.copy(),
        )
        return new

    # PRIVATE/SPECIAL METHODS =================================================

    def _gchop_h5_(self):
        """
        Extract HDF5 serialization data for this Galaxy.

        Returns metadata about the galaxy type and Astropy Tables for each
        particle type (stars, dark_matter, gas) that should be persisted.

        Returns
        -------
        metadata : dict
            Dictionary containing galaxy metadata:
            - 'galaxy_type': class name of the galaxy
            - 'has_potential': whether potential is computed
        psets : dict
            Dictionary with keys 'stars', 'dark_matter', 'gas' mapping
            to tuples of (metadata, table) for each particle set

        """
        cls = type(self)

        # Create galaxy-level metadata
        metadata = {
            "galaxy_type": cls.__name__,
            "has_potential": bool(self.has_potential_),
        }

        # Get particle set data using their _gchop_h5_ methods
        psets = {
            "stars": self.stars._gchop_h5_(),
            "dark_matter": self.dark_matter._gchop_h5_(),
            "gas": self.gas._gchop_h5_(),
        }

        return metadata, psets


# =============================================================================
# API FUNCTIONS
# =============================================================================


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
    *,
    softening_s: float = 0.0,
    softening_dm: float = 0.0,
    softening_g: float = 0.0,
    potential_s: np.ndarray = None,
    potential_dm: np.ndarray = None,
    potential_g: np.ndarray = None,
):
    """
    Galaxy builder.

    This function builds a galaxy object from a star,
    dark matter and gas ParticleSet.

    Parameters
    ----------
    m_s : np.ndarray
        Star masses. Shape: (n,1).
    x_s, y_s, z_s : np.ndarray
        Star positions. Shapes: (n,1).
    vx_s, vy_s, vz_s : np.ndarray
        Star velocities. Shape: (n,1).
    m_dm : np.ndarray
        Dark matter masses. Shape: (n,1).
    x_dm, y_dm, z_dm : np.ndarray
        Dark matter positions. Shapes: (n,1).
    vx_dm, vy_dm, vz_dm : np.ndarray
        Dark matter velocities. Shapes: (n,1).
    m_g : np.ndarray
        Gas masses. Shape: (n,1).
    x_g, y_g, z_g :  np.ndarray
        Gas positions. Shapes: (n,1).
    vx_g, vy_g, vz_g : np.ndarray
        Gas velocities. Shapes: (n,1).
    potential_s : np.ndarray, default value = None
        Specific potential energy of star particles. Shape: (n,1).
    potential_dm : np.ndarray, default value = None
        Specific potential energy of dark matter particles. Shape: (n,1).
    potential_g : np.ndarray, default value = None
        Specific potential energy of gas particles. Shape: (n,1).
    softening_s : Quantity. Default value = 0
        Softening radius of stellar particles. Shape: (1,).
        Default unit: kpc.
    softening_dm : Quantity. Default value = 0
        Softening radius of dark matter particles. Shape: (1,).
        Default unit: kpc.
    softening_g : Quantity. Default value = 0
        Softening radius of gas particles. Shape: (1,).
        Default unit: kpc.

    Return
    ------
    galaxy: ``Galaxy class`` object.

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
