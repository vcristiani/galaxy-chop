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
        super().__attrs_post_init__()
        # Validate lengths match
        if len(self) != len(self.components):
            raise ValueError(
                f"galaxy length ({len(self)}) must match "
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

    def to_dict(self, *, ptypes=None, attributes=None):
        all_dgal_attributes = [
            "components",
            "component_labels",
            "probabilities",
        ]
        gal_attributes = [a for a in attributes if a not in dgal_attributes]

        the_dict = super().to_dict(ptypes=ptypes, attributes=attributes)
        the_dict["components"] = self.components
        the_dict["component_labels"] = self.component_labels
        the_dict["probabilities"] = self.probabilities
        return the_dict

    def copy(self):
        new = super().copy()
        # we need to create a new object with the copied components
        new_dict = new.disassemble()
        new_dict["components"] = self.components.copy()
        new_dict["component_labels"] = self.component_labels.copy()
        new_dict["probabilities"] = (
            self.probabilities.copy()
            if self.probabilities is not None
            else None
        )
        return DecomposedGalaxy(**new_dict)

    def disassemble(self):
        the_dict = super().disassemble()
        the_dict["components"] = self.components
        the_dict["component_labels"] = self.component_labels.copy()
        the_dict["probabilities"] = self.probabilities
        return the_dict
