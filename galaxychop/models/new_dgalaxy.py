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

    def get_value_makers(self):
        value_makers = super().get_value_makers()
        value_makers.update(
            {
                "components": lambda: self.components.copy(),
                "labels": lambda: self.labels.copy(),
                "probabilities": lambda: self.probabilities.copy(),
            }
        )
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


@uttr.s(frozen=True, slots=True, repr=False, aaccessor=None)
class DecomposedGalaxy(Galaxy):

    method: str = uttr.ib(converter=str)
    component: np.ndarray = uttr.ib(converter=np.copy)
    component_labels: dict = uttr.ib(validator=vldt.instance_of(dict))
    probabilities: np.ndarray = uttr.ib(converter=np.copy)

    # INTERNAL ================================================================

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if len(self.method) == 0:
            raise ValueError("method cannot be empty")

        # Validate lengths match
        if len(self) != len(self.component):
            raise ValueError(
                f"galaxy length ({len(self)}) must match "
                f"component length ({len(self.component)})"
            )
        self.component.setflags(write=False)

        # Validate probabilities
        if len(self.probabilities) != len(self.component):
            raise ValueError(
                f"probabilities length ({len(self.probabilities)}) "
                f"must match component length ({len(self.component)})"
            )

        # Ensure all probability values are between 0 and 1, ignoring NaNs.
        non_nan_probs = self.probabilities[~np.isnan(self.probabilities)]
        if not np.all((non_nan_probs >= 0) & (non_nan_probs <= 1)):
            raise ValueError(
                "probabilities must be in the range [0, 1] (ignoring nans)"
            )

        self.probabilities.setflags(write=False)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""

        cls_name = type(self).__name__
        gal_repr = ", ".join(super().__repr__().split(", ")[1:-1])
        method = f"method={self.method!r}"
        component = f"component={sorted(self.unique_component_labels)}"
        probs = f"probabilities={self.has_probabilities}"

        return f"<{cls_name} {method}, {gal_repr}, {probs}, {component}>"

    # PROPERTIES ==============================================================

    @property
    def unique_component(self):
        return set(np.unique(self.component))

    @property
    def unique_component_labels(self):
        return {
            str(self.component_labels.get(component, component))
            for component in self.unique_component
        }

    @property
    def has_probabilities(self):
        return self.probabilities is not None

    # UTILITIES ===============================================================

    def label_component(self, labels=None):
        """
        Access all the labels mapped to the lmap dictionary.

        If no lmap is provided, the function tries to use the internal
        lmap dict. If the instance doesn't has an lmap dict this method
        is equivalent to access the labels attribute, but returns a copy
        with object as dtype.

        """
        lmap = self.component_labels if labels is None else lmap

        def lmapper(k):
            return lmap.get(k, k)

        return np.fromiter(map(lmapper, self.component), object)

    def to_dataframe(self, *, ptypes=None, attributes=None):

        value_makers = {
            "method": lambda: np.full(len(self), self.method),
            "component": lambda: self.component.copy(),
            "label": lambda: self.label_component(),
            "has_probabilities": lambda: (
                np.full(len(self), self.has_probabilities)
            ),
            "probabilities": lambda: (
                self.probabilities.copy()
                if self.has_probabilities
                else np.full(len(self), np.nan)
            ),
        }

        all_dgal_attributes = set(value_makers)

        if attributes is not None:

            attributes = set(attributes)

            dgal_attributes = attributes & all_dgal_attributes
            gal_attributes = attributes - all_dgal_attributes
            gal_attributes.update(["ptype", "ptypev"])

        else:
            gal_attributes = None
            dgal_attributes = all_dgal_attributes

        df = super().to_dataframe(ptypes=None, attributes=gal_attributes)

        for aname in sorted(dgal_attributes):
            mkvalue = value_makers[aname]
            avalue = mkvalue()
            avalue.setflags(write=True)
            df[aname] = avalue

        if ptypes is not None:
            ptypesv = list(map(ParticleSetType.mktype, ptypes))
            df = df[df.ptypev.isin(ptypesv)]

        if "label" in df:
            df["label"] = df["label"].fillna(df["ptype"])

        if "ptypev" not in attributes:
            df.drop("ptypev", axis=1, inplace=True)
        if "ptype" not in attributes:
            df.drop("ptype", axis=1, inplace=True)

        return df

    def copy(self):
        cls = type(self)
        new = cls(
            stars=self.stars.copy(),
            dark_matter=self.dark_matter.copy(),
            gas=self.gas.copy(),
            method=self.method,
            component=self.component,
            component_labels=self.component_labels,
            probabilities=self.probabilities,
        )
        return new

    def to_dict(self):
        # primero filtrar
        ...
