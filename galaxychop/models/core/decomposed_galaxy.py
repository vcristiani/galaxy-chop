# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Common functionalities for galaxy decomposition."""

# =============================================================================
# IMPORTS
# =============================================================================


import attr

import numpy as np

import pandas as pd

import uttr

from ...core import Galaxy, ParticleSet, ParticleSetType


# =============================================================================
# CLASSES
# =============================================================================


@uttr.s(frozen=True, slots=True, repr=False, aaccessor=None)
class DecomposedParticleSet(ParticleSet):
    """A set of particles with component information.

    This class extends `ParticleSet` to include information about the
    kinematic components to which the particles belong.

    Parameters
    ----------
    components : np.ndarray
        An array of integers identifying the component of each particle.
    labels : np.ndarray
        An array of strings with the label of the component for each particle.
    probabilities : np.ndarray
        An array of probabilities, one for each particle, indicating the
        likelihood of it belonging to its assigned component.

    """
    #: Tuple of attribute names that should NOT be serialized to HDF5 files.
    #: Extends ParticleSet.H5_TRANSIENTS with:
    #: - has_probabilities: Boolean flag stored in dataset metadata instead of
    #:   as a column, used to determine decomposition type at read time
    H5_TRANSIENTS = ParticleSet.H5_TRANSIENTS + ("has_probabilities",)

    components: np.ndarray = uttr.ib(converter=np.copy)
    labels: np.ndarray = uttr.ib(converter=lambda arr: np.astype(arr, np.str_))
    has_probabilities: bool = uttr.ib(converter=bool)
    probabilities: np.ndarray = uttr.ib(converter=np.copy)

    probabilities_n = uttr.ib(init=False)

    # CONSTRUCTORS ============================================================

    @classmethod
    def from_pset(
        cls, pset, components, labels, probabilities, has_probabilities
    ):
        """
        Create a new instance from a particle set.

        This class method extracts initialized attributes from `pset`
        using `attr.asdict`, assigns the `softening` value, and builds
        a new instance of the class with the provided components, labels,
        and probabilities.

        Parameters
        ----------
        pset : object
            A particle set object from which to extract initialized data.
            Must contain the attribute `softening`.
        components : array-like
            Component assignment for each particle.
        labels : array-like
            Labels associated with each particle.
        probabilities : array-like
            Membership probabilities of each particle across components.

        Returns
        -------
        cls
            A new instance of the class with component, label, and
            probability information.
        """
        data = attr.asdict(pset, filter=lambda a, _: a.init)

        data["softening"] = pset.softening.value

        instance = cls(
            **data,
            components=components,
            labels=labels,
            probabilities=probabilities,
            has_probabilities=has_probabilities,
        )
        return instance

    # INITIALIZATION ==========================================================

    @probabilities_n.default
    def _proabilities_n_default(self):
        probs_n = (
            np.shape(self.probabilities)[-1] if self.has_probabilities else 0
        )
        return probs_n

    def __attrs_post_init__(self):
        """
        Perform validation and adjustments after attribute initialization.

        This special method runs automatically after all attributes
        of the class have been initialized (specific to `attrs`).
        It validates the consistency between the particle array size
        and the lengths of the `components`,`labels`,and `probabilities`arrays.
        It also ensures that the arrays are read-only
        and that the probabilities are within the valid range [0, 1].

        Raises
        ------
        ValueError
            If the lengths of `components`, `labels`, or `probabilities`
            do not match the length of the particle array.
        ValueError
            If probabilities exist outside the range [0, 1] (ignoring NaNs).
        """
        super().__attrs_post_init__()

        if len(self) != len(self.components):
            raise ValueError(
                f"galaxy length ({len(self)}) must match "
                f"components length ({len(self.components)})"
            )
        self.components.setflags(write=False)

        if len(self) != len(self.labels):
            raise ValueError(
                f"galaxy length ({len(self)}) must match "
                f"labels length ({len(self.labels)})"
            )
        self.labels.setflags(write=False)

        # Validate probabilities.
        if len(self.probabilities) != len(self):
            raise ValueError(
                f"probabilities length ({len(self.probabilities)}) "
                f"must match particle set length ({len(self)})"
            )

        # Ensure all probability values are between 0 and 1, ignoring NaNs.
        non_nan_probs = self.probabilities[~np.isnan(self.probabilities)]
        invalid = non_nan_probs[(non_nan_probs < 0) | (non_nan_probs > 1)]
        if np.size(invalid):
            invalid_set = set(invalid.flatten())
            raise ValueError(
                "probabilities must be in the range [0, 1] (ignoring nans). "
                f"Found: {invalid_set}"
            )

        # Make the probabilities array read-only.
        self.probabilities.setflags(write=False)


    # PUBLIC METHODS ==========================================================

    def total_mass(self):
        """
        Calculate total mass and mass fraction for each component.

        Groups particles by their component labels and calculates
        the deterministic mass (based on assigned labels) and optionally
        the probabilistic mass (based on probability weights).

        Returns
        -------
        DataFrame : pandas DataFrame
            DataFrame with component labels as index and columns:
            - 'm': deterministic total mass in M_sun units
            - 'mf': deterministic mass fraction
            - 'mp': probabilistic total mass (if has_probabilities is True)
            - 'mpf': probabilistic mass fraction (if has_probabilities is True)

        Examples
        --------
        >>> import galaxychop as gchop
        >>> dps = gchop.models.DecomposedParticleSet(...)
        >>> dps.total_mass()
                      m        mf        mp       mpf
        bulge    3.5e9    0.250    3.4e9    0.243
        disk     6.8e9    0.486    6.9e9    0.493
        halo     3.7e9    0.264    3.7e9    0.264
        """
        # Create DataFrame with labels and masses
        df = self.to_dataframe(
            attributes=["components", "labels", "m", "probabilities"]
        )
        df["components"] = df["components"].fillna("")

        pset_total_mass = self.m.sum().value

        # Group by label and sum masses (deterministic)
        result = df.groupby(["components", "labels"])[["m"]].sum()
        result["mf"] = result["m"] / pset_total_mass

        # Add probabilistic mass columns if available
        import ipdb

        ipdb.set_trace()
        print(df.components.unique(), ((self.probabilities).shape))

        if self.has_probabilities:
            prob_masses_column = []
            prob_fraction_masses_column = []
            for component in result.index.levels[0]:
                if component != "":
                    prob_column = f"probabilities_{int(component)}"
                    cosos = df[df["components"] == component]
                    prob_mass = (cosos["m"] * cosos[prob_column]).sum()
                    prob_mass_fraction = prob_mass / pset_total_mass
                    prob_masses_column.append(prob_mass)
                    prob_fraction_masses_column.append(prob_mass_fraction)
                else:
                    prob_masses_column.append("")
                    prob_fraction_masses_column("")

        result.reset_index("labels", inplace=True)
        return result

    def get_value_makers(self):
        """
        Return value maker functions for particle set attributes.

        The result is a dictionary mapping attribute names to
        functions that return copies of the corresponding data.
        In addition to inherited makers, it includes `components`,
        `labels`, and one key for each probability column (`prob_i`).

        Returns
        -------
        dict
            A dictionary of value maker functions.
            Includes:
            - `"components"`: copy of the components array.
            - `"labels"`: copy of the labels array.
            - `"prob_i"`: copy of the i-th probability column.
        """
        value_makers = super().get_value_makers()
        component_makers = {
            "components": lambda: self.components.copy(),
            "labels": lambda: self.labels.copy(),
            "probabilities": lambda: self.probabilities.copy(),
        }

        value_makers.update(component_makers)
        return value_makers

    # CONVERSION METHODS ======================================================

    def copy(self):
        """
        Create a deep copy of the DecomposedParticleSet.

        All relevant particle set attributes are cloned, including mass,
        positions, velocities, potential, softening, components, labels,
        and probabilities.

        Returns
        -------
        DecomposedParticleSet
            A new instance identical to the original, with data copied.
        """
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
            potential=self.potential if self.has_potential_ else None,
            softening=float(self.softening.value),
            components=self.components,
            labels=self.labels,
            probabilities=self.probabilities,
            has_probabilities=self.has_probabilities,
        )
        return new

    def to_particleset(self):
        """
        Convert to a regular ParticleSet, discarding decomposition data.

        This method creates a new ParticleSet instance with the same
        physical particle data (mass, position, velocity, potential, softening)
        but without the decomposition-specific attributes (components,
        labels, probabilities).

        Returns
        -------
        ParticleSet
            A new ParticleSet instance without decomposition information.
        """
        return ParticleSet(
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

    # PRIVATE/SPECIAL METHODS =================================================

    def _gchop_h5_(self):
        """
        Extract HDF5 serialization data for this DecomposedParticleSet.

        Returns metadata about the particle set type and a DataFrame
        containing the particle data that should be persisted, including
        component information.

        Returns
        -------
        metadata : dict
            Dictionary containing particle set metadata:
            - 'pset_type': class name of the particle set
            - 'ptype': particle type (stars, dark_matter, gas)
            - 'has_potential': whether potential is computed
            - 'has_probabilities': whether probabilities are defined
            - 'probabilities_n': number of probability columns
        dataframe : pd.DataFrame
            DataFrame with particle attributes to be saved, including
            components, labels, and probabilities

        """
        # Get base metadata and table from parent
        metadata, table = super()._gchop_h5_()

        # Add component-specific metadata
        metadata.update(
            {
                "has_probabilities": bool(self.has_probabilities),
                "probabilities_n": int(self.probabilities_n),
            }
        )

        return metadata, table


# =============================================================================
# GALAXY
# =============================================================================


@uttr.s(frozen=True, slots=True, repr=False, aaccessor=None)
class DecomposedGalaxy(Galaxy):
    """
    Represent a galaxy decomposed into physical components.

    A `DecomposedGalaxy` contains stars, dark matter, and gas,
    each represented as a `DecomposedParticleSet`, along with
    labels and membership probabilities.


    Attributes
    ----------
    method : str
        The method used for the decomposition (e.g., clustering).
    component_name_mapping : dict
        Dictionary mapping component identifiers to human-readable names.

    Raises
    ------
    ValueError
        If `method` is an empty string.
    TypeError
        If any particle set is not of type `DecomposedParticleSet`.
    TypeError
        If probability configurations are inconsistent across particle sets.
    """

    PSET_CLS = DecomposedParticleSet

    method: str = uttr.ib(converter=str)
    component_name_mapping: dict = uttr.ib(converter=dict)

    # INITIALIZATION ==========================================================

    def __attrs_post_init__(self):
        """
        Validate attributes after initialization.

        Ensures that the `method` string is not empty and that all
        particle sets (`stars`, `dark_matter`, `gas`) are instances
        of `DecomposedParticleSet`. Also verifies that probability
        configurations are consistent across particle sets.

        Raises
        ------
        ValueError
            If `method` is empty.
        TypeError
            If particle sets are not of type `DecomposedParticleSet`.
        TypeError
            If probability configurations differ between particle sets.
        """
        super().__attrs_post_init__()
        if len(self.method) == 0:
            raise ValueError("method cannot be empty")

        has_probs = {}
        for pset in (self.stars, self.dark_matter, self.gas):
            has_probs[pset.ptype.name] = pset.has_probabilities

        if len(set(has_probs.values())) > 1:
            raise TypeError(
                "Inconsistent probability configurations across particle sets. "
                f"Found configurations: {has_probs}. All particle sets must have "
                "the same probability setting (all True or all False)."
            )

    def __repr__(self):
        """
        Return the string representation of the object.

        Returns
        -------
        str
            A string summarizing the method, probabilities, and
            component labels of the galaxy.
        """
        cls_name = type(self).__name__
        gal_repr = ", ".join(super().__repr__().split(", ")[1:-1])
        method = f"method={self.method!r}"
        component = f"components={sorted(self.unique_components_labels)}"
        probs = f"probabilities={self.has_probabilities}"

        return f"<{cls_name} {method}, {gal_repr}, {probs}, {component}>"

    # PROPERTIES ==============================================================

    @property
    def has_probabilities(self):
        """
        Indicate whether the galaxy particle sets include probabilities.

        Returns
        -------
        bool
            True if probabilities are defined, False otherwise.
        """
        return self.stars.has_probabilities

    @property
    def unique_components(self):
        """
        Return the unique component identifiers in the galaxy.

        Returns
        -------
        set
            A set of unique component indices.
        """
        df = self.to_dataframe(attributes=["components"])
        components_list = df["components"].unique().tolist()
        the_unique_components = set(sorted(components_list))
        return the_unique_components

    @property
    def unique_components_labels(self):
        """
        Return the unique component labels in the galaxy.

        Returns
        -------
        set
            A set of unique component labels.
        """
        df = self.to_dataframe(attributes=["labels"])
        labels_list = df["labels"].unique().tolist()
        the_unique_labels = set(sorted(labels_list))
        return the_unique_labels

    # PUBLIC METHODS ==========================================================

    def total_mass(self):
        """
        Calculate total mass and mass fraction for each component by particle type.

        Creates a hierarchical DataFrame with MultiIndex (ptype, label) containing
        the total mass and mass fraction for each component within each particle type.

        Returns
        -------
        DataFrame : pandas DataFrame
            DataFrame with MultiIndex (ptype, label) and two columns:
            - 'm': total mass in M_sun units
            - 'mf': mass fraction (component mass / particle type total mass)

        Examples
        --------
        >>> import galaxychop as gchop
        >>> dgal = gchop.models.DecomposedGalaxy(...)
        >>> dgal.total_mass()
                                     m        mf
        stars       Bulge       7.16e+09    0.189
                    Cold disk   1.63e+10    0.430
                    Halo        3.64e+09    0.096
                    Warm disk   1.02e+10    0.268
                    stars       1.40e+08    0.004
        dark_matter dark_matter 1.22e+11    1.000
        gas         gas         1.22e+11    1.000
        """
        import pandas as pd

        # Collect mass DataFrames from each particle type
        ptype_dfs = []
        for pset in [self.stars, self.dark_matter, self.gas]:
            ptype_name = pset.ptype.humanize()
            pset_mass = pset.total_mass()

            # Create MultiIndex with ptype and label levels
            pset_mass.index = pd.MultiIndex.from_product(
                [[ptype_name], pset_mass.index], names=["ptype", "components"]
            )
            ptype_dfs.append(pset_mass)

        # Concatenate all particle type DataFrames
        result = pd.concat(ptype_dfs)

        return result

    # CONVERSION METHODS ======================================================

    def copy(self):
        """
        Make a copy of the DecomposedGalaxy.

        Returns
        -------
        DecomposedGalaxy
            A new instance identical to the original, with data copied.
        """
        cls = type(self)
        new = cls(
            method=self.method,
            component_name_mapping=self.component_name_mapping.copy(),
            stars=self.stars.copy(),
            dark_matter=self.dark_matter.copy(),
            gas=self.gas.copy(),
        )
        return new

    def to_galaxy(self):
        """
        Convert to a regular Galaxy, discarding decomposition data.

        This method creates a new Galaxy instance with the same particle
        data (stars, dark matter, gas) but without the decomposition-specific
        information (method, component mappings, component labels, and
        probabilities). Each DecomposedParticleSet is converted to a regular
        ParticleSet using the to_particleset() method.

        Returns
        -------
        Galaxy
            A new Galaxy instance without decomposition information.
        """
        return Galaxy(
            stars=self.stars.to_particleset(),
            dark_matter=self.dark_matter.to_particleset(),
            gas=self.gas.to_particleset(),
        )

    # PRIVATE/SPECIAL METHODS =================================================

    def _gchop_h5_(self):
        """
        Extract HDF5 serialization data for this DecomposedGalaxy.

        Returns metadata about the decomposed galaxy and DataFrames for each
        particle type that should be persisted, including component information.

        Returns
        -------
        metadata : dict
            Dictionary containing decomposed galaxy metadata:
            - 'galaxy_type': class name of the galaxy
            - 'has_potential': whether potential is computed
            - 'pset_types': metadata for each particle set type
            - 'method': decomposition method used
            - 'component_name_mapping': mapping of component IDs to names
            - 'has_probabilities': whether probabilities are defined
        dataframes : dict
            Dictionary with keys 'stars', 'dark_matter', 'gas' mapping
            to their respective DataFrames with component information

        """
        # Get base metadata and dataframes from parent
        metadata, dataframes = super()._gchop_h5_()

        # Add decomposed galaxy specific metadata
        metadata.update(
            {
                "method": self.method,
                "component_name_mapping": self.component_name_mapping,
                "has_probabilities": bool(self.has_probabilities),
            }
        )

        return metadata, dataframes
