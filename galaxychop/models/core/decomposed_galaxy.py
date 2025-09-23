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

import functools
from collections import OrderedDict


import attr
from attr import validators as vldt

import numpy as np

import pandas as pd

import uttr


from ..core.data import Galaxy, mkgalaxy, ParticleSetType, ParticleSet


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
        # This method is called after all attributes are initialized.
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
        return bool(self.probabilities_n)

    def get_value_makers(self):
        value_makers = super().get_value_makers()
        component_makers = {
            "components": lambda: self.components.copy(),
            "labels": lambda: self.labels.copy(),
        }

        for n in range(self.probabilities_n):
            prob_maker = lambda: self.probabilities[:, n].copy()
            component_makers[f"prob_{n}"] = prob_maker

        value_makers.update(component_makers)
        return value_makers

    def copy(self):
        "Make a copy of the ComponentParticleSet."
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

    method: str = uttr.ib(converter=str)
    component_name_mapping: dict = uttr.ib(converter=dict)

    # INTERNAL ================================================================

    def __attrs_post_init__(self):
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
                "Inconsistent probability configurations across particle sets. "
                f"Found configurations: {has_probs}. All particle sets must have "
                "the same probability setting (all True or all False)."
            )

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""

        cls_name = type(self).__name__
        gal_repr = ", ".join(super().__repr__().split(", ")[1:-1])
        method = f"method={self.method!r}"
        component = f"components={sorted(self.unique_components_labels)}"
        probs = f"probabilities={self.has_probabilities}"

        return f"<{cls_name} {method}, {gal_repr}, {probs}, {component}>"

    # PROPERTIES ==============================================================

    @property
    def has_probabilities(self):
        return self.stars.has_probabilities

    @property
    def unique_components(self):
        df = self.to_dataframe(attributes=["components"])
        components_list = df["components"].unique().tolist()
        the_unique_components = set(sorted(components_list))
        return the_unique_components

    @property
    def unique_components_labels(self):
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
