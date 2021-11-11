# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Common functionalities for galaxy decomposition."""


# =============================================================================
# IMPORTS
# =============================================================================

import abc

import attr
from attr import validators as vldt

import numpy as np

import pandas as pd

from .. import data, utils

# =============================================================================
# CONSTANTS
# =============================================================================

_CIRCULARITY_ATTRIBUTES = ("normalized_star_energy", "eps", "eps_r")

_PTYPES_ORDER = tuple(p.name.lower() for p in data.ParticleSetType)


# =============================================================================
# RESULT
# =============================================================================


@attr.s(frozen=True, slots=True, repr=False)
class _Components:
    labels = attr.ib(validator=vldt.instance_of(np.ndarray))
    ptypes = attr.ib(validator=vldt.instance_of(np.ndarray))
    probabilities = attr.ib(
        validator=vldt.optional(vldt.instance_of(np.ndarray))
    )

    def __attrs_post_init__(self):
        lens = {len(self.labels), len(self.ptypes)}
        if self.probabilities is not None:
            lens.add(len(self.probabilities))
        if len(lens) > 1:
            raise ValueError("All length must be the same")

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        length = len(self)
        labels = np.unique(self.labels)
        probs = self.probabilities is not None
        return f"components({length}labels={labels}, probabilities={probs})"


# =============================================================================
# FUNCTIONS
# =============================================================================


def hparam(default, **kwargs):
    metadata = kwargs.pop("metadata", {})
    metadata["__gchop_model_hparam__"] = True
    return attr.ib(default=default, metadata=metadata, kw_only=True, **kwargs)


# =============================================================================
# ABC
# =============================================================================
@attr.s(frozen=True, repr=False)
class GalaxyDecomposerABC(metaclass=abc.ABCMeta):

    __gchop_model_cls_config__ = {"repr": False, "frozen": True}

    cbins = hparam(default=(0.05, 0.005))

    @cbins.validator
    def _bins_validator(self, attribute, value):
        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], float)
            and isinstance(value[1], float)
        ):
            raise ValueError("cbins must be a tuple of two floats.")

    # block meta checks =======================================================
    def __init_subclass__(cls):
        model_config = getattr(cls, "__gchop_model_cls_config__")
        attr.s(maybe_cls=cls, **model_config)

    # block  to implement in every method =====================================

    @abc.abstractmethod
    def get_attributes(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_rows_mask(self, X, y, attributes):
        raise NotImplementedError()

    @abc.abstractmethod
    def split(self, X, y, attributes):
        raise NotImplementedError()

    # internal ================================================================

    def __repr__(self):
        clsname = type(self).__name__

        selfd = attr.asdict(
            self,
            recurse=False,
            filter=lambda attr, _: attr.repr,
        )
        attrs_str = ", ".join([f"{k}={repr(v)}" for k, v in selfd.items()])
        return f"{clsname}({attrs_str})"

    # API =====================================================================

    def _get_jcirc_df(self, galaxy, attributes):

        # STARS
        # turn the galaxy into jcirc dict
        # all the calculation cames together so we can't optimize here
        jcirc = utils.jcirc(galaxy, *self.cbins)._asdict()

        # we add the colum with the types, all the values from jcirc
        # are stars
        jcirc["ptypev"] = data.ParticleSetType.STARS.value
        stars_df = pd.DataFrame({attr: jcirc[attr] for attr in attributes})

        # DARK_MATTER
        dm_rows = len(galaxy.dark_matter)
        dm_nans = np.full(dm_rows, np.nan)

        dm_columns = {attr: dm_nans for attr in attributes}
        dm_columns["ptypev"] = data.ParticleSetType.DARK_MATTER.value

        dm_df = pd.DataFrame(dm_columns)

        # GAS
        gas_rows = len(galaxy.gas)
        gas_nans = np.full(gas_rows, np.nan)

        gas_columns = {attr: gas_nans for attr in attributes}
        gas_columns["ptypev"] = data.ParticleSetType.GAS.value

        gas_df = pd.DataFrame(gas_columns)

        return pd.concat([stars_df, dm_df, gas_df], ignore_index=True)

    def attributes_matrix(self, galaxy, attributes):
        """Retorna 2 elementos:

        - Un numpy array X de 2d en el cual cada fila representa una
          particula, y cada columna un attributo
        - Un array 'y' con la misma longitud que filas de X que representa
          el tipo de particula en cada fila
          (0 = STARS, 1=DM, 2=Gas)

        Tiene que haber tantas filas como el total de particulas de la galaxia.

        """
        # first we split the attributes between the ones from circularity
        # and the ones from "galaxy.to_dataframe()"
        circ_attrs, df_attrs = [], []
        for attr_name in attributes:
            container = (
                circ_attrs
                if attr_name in _CIRCULARITY_ATTRIBUTES
                else df_attrs
            )
            container.append(attr_name)

        # this crap is going to have all the dataframes that contain as a
        # column each attribute
        result = []

        # If we have attributes of "to_dataframe" =============================
        #     now we take out all the attributes of "to_dataframe" and save
        #     them in a list where all the resulting dataframes will be stored
        if df_attrs:

            # we need this to create the array of classes
            if "ptypev" not in df_attrs:
                df_attrs.append("ptypev")

            dfgal = galaxy.to_dataframe(
                ptypes=_PTYPES_ORDER, attributes=df_attrs
            )
            result.append(dfgal)

        # If we have JCIRC attributes =========================================
        #     I'm going to need a lot of NANs that represent that gas and dm
        #     have no circularity.
        if circ_attrs:
            circ_attrs.append("ptypev")
            dfcirc = self._get_jcirc_df(galaxy, circ_attrs)
            result.append(dfcirc)

        # the attributes as dataframe
        df = pd.concat(result, axis=1)

        # remove if ptypev is duplicated
        df = df.loc[:, ~df.columns.duplicated()]

        # separamos la matriz y las clases
        X = df[attributes].to_numpy()
        y = df.ptypev.to_numpy()

        # retornamos
        return X, y

    def complete_labels(self, X, labels, rows_mask):
        new_labels = np.full(len(X), np.nan)
        new_labels[rows_mask] = labels
        return new_labels

    def complete_probs(self, X, probs, rows_mask):
        if probs is None:
            return None

        # the number of particles are incorrect so we simple remove the data
        probs_shape = list(np.shape(probs)[1:])

        # We need this many rows
        complete_shape = tuple([len(X)] + probs_shape)

        # now we create the container for the probabilities
        new_probs = np.full(complete_shape, np.nan)

        # and now we inject the probs in the correct order
        new_probs[rows_mask] = probs

        return new_probs

    def decompose(self, galaxy):
        """Decompose method.

        Assign the component of the galaxy to which each particle belongs.
        Validation of the input galaxy instance.

        Parameters
        ----------
        galaxy :
            `galaxy object`
        """
        attributes = self.get_attributes()

        X, y = self.attributes_matrix(galaxy, attributes=attributes)

        # calculate only the valid values to operate the clustering
        rows_mask = self.get_rows_mask(X=X, y=y, attributes=attributes)
        X_clean, y_clean = X[rows_mask], y[rows_mask]

        # execute the cluster with the quantities of interest
        labels, probs = self.split(X=X_clean, y=y_clean, attributes=attributes)

        # retrieve and fix the labels
        final_labels = self.complete_labels(
            X=X, labels=labels, rows_mask=rows_mask
        )
        final_probs = self.complete_probs(
            X=X, probs=probs, rows_mask=rows_mask
        )
        final_y = np.array(
            [data.ParticleSetType.mktype(yi).humanize() for yi in y]
        )

        # return the instance
        return _Components(
            labels=final_labels,
            ptypes=final_y,
            probabilities=final_probs,
        )


# =============================================================================
# MIXIN
# =============================================================================


def get_attributes(self):
    return ["normalized_star_energy", "eps", "eps_r"]


class DynamicStarsDecomposerMixin:
    def get_rows_mask(self, X, y, attributes):
        # all the rows where every value is finite
        only_stars = np.equal(y, data.ParticleSetType.STARS.value)
        finite_values = np.isfinite(X).all(axis=1)
        return only_stars & finite_values
