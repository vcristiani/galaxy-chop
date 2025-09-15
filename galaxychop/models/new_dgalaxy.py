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


from attr import validators as vldt

import numpy as np

import pandas as pd

import uttr


from ..core.data import Galaxy


# =============================================================================
# CLASSES
# =============================================================================


@uttr.s(frozen=True, slots=True, repr=False, aaccessor=None)
class DecomposedGalaxy(Galaxy):

    # Decomposed galaxy attributes
    components: np.ndarray = uttr.ib(converter=np.copy)
    component_labels: dict = uttr.ib(validator=vldt.instance_of(dict))
    probabilities: np.ndarray = uttr.ib(
        default=None,
        converter=lambda v: np.copy(v) if v is not None else v,
    )

    def __attrs_post_init__(self):
        # Validate lengths match
        if len(self.galaxy) != len(self.components):
            raise ValueError(
                f"galaxy length ({len(self.galaxy)}) must match "
                f"components length ({len(self.components)})"
            )

        # Validate probabilities
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.components):
                raise ValueError(
                    f"probabilities length ({len(self.probabilities)}) "
                    f"must match components length ({len(self.components)})"
                )

            if not np.all(
                (self.probabilities >= 0) & (self.probabilities <= 1)
            ):
                raise ValueError("probabilities must be in range [0, 1]")

        # Validate component_labels has all necessary keys
        unique_components = np.unique(self.components)
        missing_keys = set(unique_components) - set(
            self.component_labels.keys()
        )
        if missing_keys:
            raise ValueError(
                "component_labels is missing keys "
                f"for components: {missing_keys}"
            )

        # Make arrays read-only like in ParticleSet
        self.components.setflags(write=False)
        if self.probabilities is not None:
            self.probabilities.setflags(write=False)

    def __getattr__(self, a):
        return getattr(self.pset, a)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        n_components = len(self.unique_components)
        has_probs = self.probabilities is not None
        return (
            f"<DecomposedGalaxy n_components={n_components}, "
            f"has_probs={has_probs}>"
        )

    def __len__(self):
        return len(self.pset)

    @property
    def unique_components(self):
        """Get unique components in the component set."""
        return np.unique(self.components)

    def get_particles_by_component(self, component):
        """
        Get particles with specific component.

        Parameters
        ----------
        component : int or str
            Component to filter by.

        Returns
        -------
        ComponentParticleSet
            New instance with only particles matching the component.
        """
        mask = self.components == component

        # Create filtered ParticleSet
        filtered_pset = ParticleSet(
            ptype=self.pset.ptype,
            m=self.pset.m[mask],
            x=self.pset.x[mask],
            y=self.pset.y[mask],
            z=self.pset.z[mask],
            vx=self.pset.vx[mask],
            vy=self.pset.vy[mask],
            vz=self.pset.vz[mask],
            potential=(
                self.pset.potential[mask] if self.pset.has_potential_ else None
            ),
            softening=self.pset.softening,
        )

        filtered_probs = (
            self.probabilities[mask]
            if self.probabilities is not None
            else None
        )

        return ComponentParticleSet(
            pset=filtered_pset,
            components=self.components[mask],
            component_labels=self.component_labels,
            probabilities=filtered_probs,
        )

    def get_component_label(self, component):
        """
        Get human-readable label for component.

        Parameters
        ----------
        component : int or str
            Component to get label for.

        Returns
        -------
        str
            Human-readable label for the component.
        """
        return self.component_labels.get(component)

    def component_counts(self):
        """
        Get count of particles per component.

        Returns
        -------
        dict
            Dictionary mapping component -> count.
        """
        unique, counts = np.unique(self.components, return_counts=True)
        return dict(zip(unique, counts))

    def to_dict(self, *, attributes=None):
        """
        Convert to dictionary including components and probabilities.

        Parameters
        ----------
        attributes : tuple, optional
            Attributes to include from the ParticleSet.

        Returns
        -------
        dict
            Dictionary with all data including components and probabilities.
        """
        # Get base ParticleSet dict
        the_dict = self.pset.to_dict(attributes=attributes)

        # Add components
        the_dict["components"] = self.components.copy()
        the_dict["components"].setflags(write=True)

        # Add probabilities if available
        if self.probabilities is not None:
            the_dict["probabilities"] = self.probabilities.copy()
            the_dict["probabilities"].setflags(write=True)
        else:
            the_dict["probabilities"] = np.full(len(self), np.nan)

        # Add component labels
        component_names = np.array(
            [self.component_labels[comp] for comp in self.components]
        )
        the_dict["component_names"] = component_names

        return the_dict

    def to_dataframe(self, *, attributes=None):
        """
        Convert to pandas DataFrame including components and probabilities.

        Parameters
        ----------
        attributes : tuple, optional
            Attributes to include from the ParticleSet.

        Returns
        -------
        DataFrame
            pandas DataFrame with all data.
        """
        the_dict = self.to_dict(attributes=attributes)
        return pd.DataFrame(the_dict)

    def copy(self):
        """
        Make a copy of the ComponentParticleSet.

        Returns
        -------
        ComponentParticleSet
            New instance with copied data.
        """
        probabilities = (
            self.probabilities.copy()
            if self.probabilities is not None
            else None
        )

        return ComponentParticleSet(
            pset=self.pset.copy(),
            components=self.components.copy(),
            component_labels=self.component_labels.copy(),
            probabilities=probabilities,
        )


class DecomposedGalaxy:
    def __init__(self, galaxy, components):
        self.galaxy = galaxy
        self.components = components
