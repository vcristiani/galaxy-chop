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

import uttr


from ...core import Galaxy, ParticleSet, ParticleSetType


# =============================================================================
# CLASSES
# =============================================================================


@uttr.s(frozen=True, slots=True, repr=False, aaccessor=None)
class ComponentParticleSet(ParticleSet):
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

    components: np.ndarray = uttr.ib(converter=np.copy)
    labels: np.ndarray = uttr.ib(converter=lambda arr: np.astype(arr, np.str_))
    probabilities: np.ndarray = uttr.ib(converter=np.copy)
    probabilities_n = uttr.ib(init=False)

    @classmethod
    def from_pset(
        cls,
        pset,
        components,
        labels,
        probabilities,
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
        )
        return instance

    @probabilities_n.default
    def _proabilities_n_default(self):
        probs_n = np.shape(self.probabilities)[-1]
        return probs_n - 1

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

    @property
    def has_probabilities(self):
        """
        Indicates whether the set of particles has associated probabilities.

        Returns
        -------
        bool
            `True` if probabilities exist (`probabilities_n > 0`),
            `False` otherwise.
        """
        return bool(self.probabilities_n)

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

    def copy(self):
        """
        Create a deep copy of the ComponentParticleSet.

        All relevant particle set attributes are cloned, including mass,
        positions, velocities, potential, softening, components, labels,
        and probabilities.

        Returns
        -------
        ComponentParticleSet
            A new instance identical to the original, with data copied.
        """
        cls = type(self)
        new = cls(
            ptype=self.ptype,
            m=self.m.copy(),
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            vx=self.vx.copy(),
            vy=self.vy.copy(),
            vz=self.vz.copy(),
            potential=self.potential.copy() if self.has_potential_ else None,
            softening=float(self.softening.value),
            components=self.components.copy(),
            labels=self.labels.copy(),
            probabilities=self.probabilities.copy(),
        )
        return new

    def _gchop_h5_(self):
        """
        Extract HDF5 serialization data for this ComponentParticleSet.

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
        # Get base metadata and dataframe from parent
        metadata, table = super()._gchop_h5_()

        # Add component-specific metadata
        metadata.update(
            {
                "has_probabilities": self.has_probabilities,
                "probabilities_n": self.probabilities_n,
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
    each represented as a `ComponentParticleSet`, along with
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
        If any particle set is not of type `ComponentParticleSet`.
    TypeError
        If probability configurations are inconsistent across particle sets.
    """

    PSET_CLS = ComponentParticleSet

    method: str = uttr.ib(converter=str)
    component_name_mapping: dict = uttr.ib(converter=dict)

    # INTERNAL ================================================================

    def __attrs_post_init__(self):
        """
        Validate attributes after initialization.

        Ensures that the `method` string is not empty and that all
        particle sets (`stars`, `dark_matter`, `gas`) are instances
        of `ComponentParticleSet`. Also verifies that probability
        configurations are consistent across particle sets.

        Raises
        ------
        ValueError
            If `method` is empty.
        TypeError
            If particle sets are not of type `ComponentParticleSet`.
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

    # REDEFINE ================================================================

    def copy(self):
        """Make a copy of the Galaxy."""
        cls = type(self)
        new = cls(
            method=self.method,
            component_name_mapping=self.component_name_mapping.copy(),
            stars=self.stars.copy(),
            dark_matter=self.dark_matter.copy(),
            gas=self.gas.copy(),
        )
        return new

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
                "has_probabilities": self.has_probabilities,
            }
        )

        return metadata, dataframes


# =============================================================================
# API FUNCTIONS
# =============================================================================


def mkdgalaxy(
    method: str,
    component_name_mapping: dict,
    m_s: np.ndarray,
    x_s: np.ndarray,
    y_s: np.ndarray,
    z_s: np.ndarray,
    vx_s: np.ndarray,
    vy_s: np.ndarray,
    vz_s: np.ndarray,
    components_s: np.ndarray,
    labels_s: np.ndarray,
    probabilities_s: np.ndarray,
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
    Decomposed galaxy builder.

    This function builds a decomposed galaxy object from star,
    dark matter and gas ComponentParticleSet. Only stellar particles
    have meaningful component assignments; dark matter and gas particles
    are assigned NaN values for components, labels, and probabilities.

    Parameters
    ----------
    m_s : np.ndarray
        Star masses. Shape: (n,1).
    x_s, y_s, z_s : np.ndarray
        Star positions. Shapes: (n,1).
    vx_s, vy_s, vz_s : np.ndarray
        Star velocities. Shape: (n,1).
    components_s : np.ndarray
        Star component identifiers. Shape: (n,1).
    labels_s : np.ndarray
        Star component labels. Shape: (n,1).
    probabilities_s : np.ndarray
        Star component probabilities. Shape: (n, n_components) or (n,1).
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
    method : str
        The decomposition method used.
    component_name_mapping : dict
        Dictionary mapping component identifiers to human-readable names.
    potential_s : np.ndarray, default value = None
        Specific potential energy of star particles. Shape: (n,1).
    potential_dm : np.ndarray, default value = None
        Specific potential energy of dark matter particles. Shape: (n,1).
    potential_g : np.ndarray, default value = None
        Specific potential energy of gas particles. Shape: (n,1).
    softening_s : float. Default value = 0
        Softening radius of stellar particles. Shape: (1,).
        Default unit: kpc.
    softening_dm : float. Default value = 0
        Softening radius of dark matter particles. Shape: (1,).
        Default unit: kpc.
    softening_g : float. Default value = 0
        Softening radius of gas particles. Shape: (1,).
        Default unit: kpc.

    Return
    ------
    decomposed_galaxy: ``DecomposedGalaxy`` object.

    """
    # Create stellar ComponentParticleSet with actual decomposition data
    stars = ComponentParticleSet(
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
        components=components_s,
        labels=labels_s,
        probabilities=probabilities_s,
    )

    # Create dark matter ComponentParticleSet with NaN component data
    # following the pattern in galaxy_decomposer_abc.py:251-257
    name_dm = ParticleSetType.DARK_MATTER.humanize()
    count_dm = len(m_dm)
    prob_shape_dm = probabilities_s.shape[1] if probabilities_s.ndim > 1 else 1

    dark_matter = ComponentParticleSet(
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
        components=np.full(count_dm, np.nan),
        labels=np.full(count_dm, name_dm, dtype=object),
        probabilities=np.full((count_dm, prob_shape_dm), np.nan),
    )
    ParticleSetType.DARK_MATTER.humanize()

    # Create gas ComponentParticleSet with NaN component data
    # following the pattern in galaxy_decomposer_abc.py:259-266
    name_gas = ParticleSetType.GAS.humanize()
    count_gas = len(m_g)

    gas = ComponentParticleSet(
        ptype=ParticleSetType.GAS,
        m=m_g,
        x=x_g,
        y=y_g,
        z=z_g,
        vx=vx_g,
        vy=vy_g,
        vz=vz_g,
        softening=softening_g,
        potential=potential_g,
        components=np.full(count_gas, np.nan),
        labels=np.full(count_gas, name_gas, dtype=object),
        probabilities=np.full((count_gas, prob_shape_dm), np.nan),
    )

    decomposed_galaxy = DecomposedGalaxy(
        method=method,
        component_name_mapping=component_name_mapping,
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )
    return decomposed_galaxy
