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


from attr import validators as vldt

import numpy as np

import pandas as pd

import uttr


from ..core.data import Galaxy, mkgalaxy


# =============================================================================
# CLASSES
# =============================================================================


@uttr.s(frozen=True, slots=True, repr=False)
class DecomposedGalaxy:

    # Decomposed galaxy attributes
    galaxy: Galaxy = uttr.ib(validator=vldt.instance_of(Galaxy))

    method: str = uttr.ib(converter=str)
    components: np.ndarray = uttr.ib(converter=np.copy)
    component_labels: dict = uttr.ib(validator=vldt.instance_of(dict))
    probabilities: np.ndarray = uttr.ib(
        default=None,
        converter=lambda v: np.copy(v) if v is not None else v,
    )

    def __attrs_post_init__(self):
        if len(self.method) == 0:
            raise ValueError("method cannot be empty")

        # Validate lengths match
        if len(self.galaxy) != len(self.components):
            raise ValueError(
                f"galaxy length ({len(self)}) must match "
                f"components length ({len(self.components)})"
            )
        self.components.setflags(write=False)

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

            self.probabilities.setflags(write=False)

    @property
    def unique_components(self):
        return set(np.unique(self.components))

    @property
    def unique_components_labels(self):
        return {
            str(self.component_labels.get(component, component))
            for component in self.unique_components
        }

    @property
    def has_probabilities(self):
        return self.probabilities is not None

    def __len__(self):
        return len(self.components)

    def __getattr__(self, a):
        """x.__getattr__(y) <==> x.y."""
        return getattr(self.galaxy, a)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""

        cls_name = type(self).__name__
        gal_repr = ", ".join(repr(self.galaxy).split(", ")[1:-1])
        method = f"method={self.method!r}"
        components = f"components={self.unique_components_labels}"
        probs = f"probabilities={self.has_probabilities}"

        return f"<{cls_name} {method}, {gal_repr}, {probs}, {components}>"

    def copy(self):
        return self.__class__(
            galaxy=self.galaxy,
            method=self.method,
            components=self.components,
            component_labels=self.component_labels,
            probabilities=self.probabilities,
        )
