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

# import functools
# from collections import OrderedDict


import attr
# from attr import validators as vldt

import numpy as np

# import pandas as pd

import uttr


from ...core import Galaxy, ParticleSet


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
    labels: np.ndarray = uttr.ib(converter=np.copy)
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
        if not np.all((non_nan_probs >= 0) & (non_nan_probs <= 1)):
            raise ValueError(
                "probabilities must be in the range [0, 1] (ignoring nans)"
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
        }

        for n in range(self.probabilities_n):
            def prob_maker(n=n):
                return self.probabilities[:, n].copy()
            component_makers[f"prob_{n}"] = prob_maker

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


# =============================================================================
# GALAXY
# =============================================================================


@uttr.s(frozen=True, slots=True, repr=False, aaccessor=None)
class DecomposedGalaxy(Galaxy):
    """
    Represent a galaxy decomposed into physical components.

    Represent a galaxy decompos
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
            if not isinstance(pset, ComponentParticleSet):
                raise TypeError(
                    f"particle set {pset!r} "
                    "must be of type ComponentParticleSet"
                )
            has_probs[pset.ptype.name] = pset.has_probabilities

        if len(set(has_probs.values())) > 1:
            raise TypeError(
                "Inconsistent probability configurations across particle sets"
                f"Found configurations:{has_probs}.All particle sets must have"
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
