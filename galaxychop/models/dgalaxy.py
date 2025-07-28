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

from collections import OrderedDict

import attr
from attr import validators as vldt

import numpy as np

import pandas as pd

import uttr

from .. import core

# =============================================================================
# RESULT
# =============================================================================


@attr.s(frozen=True, slots=True, repr=False)
class Components:
    """
    Class of components resulting from dynamic decomposition.

    This class creates the components of the galaxy from the result of the
    dynamic decomposition.

    Parameters
    ----------
    labels : np.ndarray
        1D array with the index of the component to which each particle
        belongs. Shape: (n,1).
    ptypes : np.ndarray
        Indicates the type of particle: stars = 0, dark matter = 1, gas = 2.
        Shape: (n,1).
    m : np.ndarray
        Particle masses. Shape: (n,1).
    lmap : dict
        Meaning of the component numbers.
    probabilities : np.ndarray or None
       1D array with probabilities of the particles to belong to each
       component, in case the dynamic decomposition model includes them.
       Shape: (n,1).
       Otherwise it adopts the value None.
    """

    labels = attr.ib(validator=vldt.instance_of(np.ndarray))
    ptypes = attr.ib(validator=vldt.instance_of(np.ndarray))
    m = attr.ib(validator=vldt.instance_of(np.ndarray))
    lmap = attr.ib(validator=vldt.instance_of(dict))
    probabilities = attr.ib(
        validator=vldt.optional(vldt.instance_of(np.ndarray))
    )

    def __attrs_post_init__(self):
        """
        Length validator.

        This method validates that the lengths of labels, ptypes are equal.
        On the other hand, if probabilities is not None, its length must be the
        same as ptypes and labels.

        """
        lens = {len(self.labels), len(self.ptypes), len(self.m)}
        if self.probabilities is not None:
            lens.add(len(self.probabilities))
        if len(lens) > 1:
            raise ValueError("All length must be the same")

    def map_labels(self, lmap=None):
        """
        Access all the labels mapped to the lmap dictionary.

        If no lmap is provided, the function tries to use the internal
        lmap dict. If the instance doesn't has an lmap dict this method
        is equivalent to access the labels attribute, but returns a copy
        with object as dtype.

        """
        lmap = self.lmap if lmap is None else lmap

        def lmapper(k):
            return lmap.get(k, k)

        return np.fromiter(map(lmapper, self.labels), object)

    def __len__(self):
        """x.__len__() <==> len(x)."""
        return len(self.labels)

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        length = len(self)
        labels = sorted(
            {
                str(self.lmap.get(label, label))
                for label in np.unique(self.labels)
            }
        )
        lmap = bool(self.lmap)
        probs = True if self.probabilities is not None else False

        return (
            f"<Components length={length}, labels={labels}, "
            + f"probabilities={probs}, lmap={lmap}>"
        )

    def to_dataframe(self, attributes=None, lmap=None):
        """
        Convert to pandas data frame.

        This method builds a data frame of all parameters of Components.

        Return
        ------
        DataFrame : pandas.DataFrame
            DataFrame of all Components data.

        """
        columns_makers = {
            "m": lambda: self.m,
            "labels": lambda: self.labels,
            "ptypes": lambda: self.ptypes,
            "lmap": lambda: self.map_labels(lmap=lmap),
        }

        default_attributes = list(columns_makers) + ["probabilities"]

        attributes = default_attributes if attributes is None else attributes

        data = OrderedDict()
        probs_df = None
        for aname in attributes:
            if aname == "probabilities":
                if self.probabilities is not None:
                    probs_df = pd.DataFrame(self.probabilities)
                    probs_df.columns = [f"probs_{c}" for c in probs_df.columns]
            else:
                mkcolumn = columns_makers[aname]
                data[aname] = mkcolumn()

        df = pd.DataFrame(data)
        if probs_df is not None:
            df = pd.concat([df, probs_df], axis=1)

        return df

    def describe(self, lmap=None):
        """
        Create a description of the sizes and masses of each component.

        The method takes into account only stellar particles that could be
        classified.

        Parameters
        ----------
        lmap: dict or None, default None
            Meaning of the component numbers.
            Converts each component label to the mapped value. By
            default uses the ones provided by the decomposer.

        Returns
        -------
        pandas.DataFrame
            Information regarding component sizes and masses.

        """
        labeled_df = self.to_dataframe()
        labeled_df = labeled_df[~pd.isna(labeled_df.labels)]

        del labeled_df["ptypes"]

        total_size, total_mass = len(labeled_df), labeled_df.m.sum()
        has_probs = self.probabilities is not None

        if has_probs:
            # We create a dict that make the relation label -> prob_column
            probs_column_map = {
                int(label): f"probs_{int(label)}"
                for label in labeled_df.labels.unique()
            }

            # multiply every probability by the mass
            probs_columns = list(probs_column_map.values())
            probs_m_particles = labeled_df[probs_columns].apply(
                lambda col: (col * labeled_df["m"])
            )

            # add all the mass_prob and convert to a dict
            # {"probs_0": X.xxx, "probs_1": Y.yyy}
            # where X.xxx and Y.yyy are the mass probability
            probs_m = probs_m_particles.sum().to_dict()

            # cleanup
            del probs_columns, probs_m_particles

        components, rows = list(labeled_df.labels.unique().astype(int)), []
        components.sort()

        for component_label in components:
            component = labeled_df[labeled_df.labels == component_label]

            row = OrderedDict()

            row[("Particles", "Size")] = len(component)
            row[("Particles", "Fraction")] = len(component) / total_size

            component_mass = component.m.sum()
            row[("Deterministic mass", "Size")] = component_mass
            row[("Deterministic mass", "Fraction")] = (
                component_mass / total_mass
            )

            if has_probs:
                probs_m_column = probs_column_map[component_label]
                component_mass_fuss = probs_m[probs_m_column]

                row[("Probabilistic mass", "Size")] = component_mass_fuss
                row[("Probabilistic mass", "Fraction")] = (
                    component_mass_fuss / total_mass
                )

            rows.append(row)

        lmap = self.lmap if lmap is None else lmap
        components = [lmap.get(c, c) for c in components]

        describe_df = pd.DataFrame(rows, index=components, columns=row.keys())

        return describe_df


# =============================================================================
# DECOMPOSEDGALAXY CLASS
# =============================================================================


@uttr.s(frozen=True, repr=False)
class DecomposedGalaxy:
    """
    DecomposedGalaxy class.

    Builds an object from a ``Galaxy`` and its ``Components`` obtained
    after applying a dynamical decomposition method to it.

    Parameters
    ----------
    Galaxy : ``Galaxy``
        Instance of ``Galaxy``.
    Component : ``Component``
        Instance of ``Component``.

    Attributes
    ----------
    WIP

    """

    galaxy = uttr.ib(validator=attr.validators.instance_of(core.data.Galaxy))
    components = uttr.ib(validator=attr.validators.instance_of(Components))

    def __len__(self):
        """len(x) <=> x.__len__()."""
        return len(self.galaxy)

    def __repr__(self):
        """repr(x) <=> x.__repr__()."""
        return repr(self.galaxy) + "\n" + repr(self.components)
