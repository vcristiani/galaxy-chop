# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Common functionalities for galaxy decomposition."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
from numpy.core.shape_base import block

from .. import data, utils

# =============================================================================
# BASE MIXIN
# =============================================================================


class GalaxyDecomposeMixin:
    """GalaxyChop decompose mixin class.

    Implementation of the particle decomposition as a
    method of the class.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of stellar particles.
        Index of the cluster each stellar particles belongs to.
    """

    def get_clean_mask(self, X):
        """Clean the Nan value of the circular parameter array.

        Parameters
        ----------
        X : `np.ndarray(n,10)`
            2D array where each file it is a different stellar particle and
            each column is a parameter of the particles:
            (m_s, x_s, y_s, z_s, vx_s, vy_s, vz_z, E_s, eps_s, eps_r_s)

        Return
        ------
        clean: np.ndarray(n_m: number of particles with E<=0 and -1<eps<1).
            Mask: Index of valid stellar particles to operate the clustering.
        """
        eps = X[:, data.Columns.eps.value]
        (clean,) = np.where(~np.isnan(eps))
        return clean

    def label_dirty(self, X, labels, clean_mask):
        eps = X[:, data.Columns.eps.value]
        complete = -np.ones(len(eps), dtype=int)
        complete[clean_mask] = labels
        return complete

    def prepare_values(self, galaxy):
        paramcirc = galaxy.paramcirc

        stars = galaxy.stars.as_array()
        stars = np.delete(stars, np.s_[7:], axis=1)

        n = len(stars)

        X = np.hstack(
            (
                stars,
                paramcirc[0].reshape(
                    n, 1
                ),  # esto es los 3 primeros del jscirc
                paramcirc[1].reshape(n, 1),
                paramcirc[2].reshape(n, 1),
            )
        )
        y = np.zeros(n, dtype=int)
        return X, y

    def get_columns(self):
        """Obtain the columns of the quantities to be used.

        Returns
        -------
        columns: list
            Only the needed columns used to decompose galaxies.

        """
        # m = 0
        # """Masses"""
        # x = 1
        # """x-position"""
        # y = 2
        # """y-position"""
        # z = 3
        # """z-position"""
        # vx = 4
        # """x-component of velocity"""
        # vy = 5
        # """y-component of velocity"""
        # vz = 6
        # """z-component of velocity"""
        # normalized_energy = 7
        # """Normalized specific energy of stars"""
        # eps = 8
        # """Circularity param"""
        # eps_r = 9
        # """Circularity param r"""

        return [
            data.Columns.normalized_energy.value,
            data.Columns.eps.value,
            data.Columns.eps_r.value,
        ]

    def decompose(self, galaxy):
        """Decompose method.

        Assign the component of the galaxy to which each particle belongs.
        Validation of the input galaxy instance.

        Parameters
        ----------
        galaxy :
            `galaxy object`
        """
        if not isinstance(galaxy, data.Galaxy):
            found = type(galaxy)
            raise TypeError(
                f"'galaxy' must be a data.Galaxy instance. Found {found}"
            )

        # retrieve te galaxy as an array os star particles
        X, y = self.prepare_values(galaxy)

        # calculate only the valid values to operate the clustering
        clean_mask = self.get_clean_mask(X)
        X_clean, y_clean = X[clean_mask], y[clean_mask]

        # select only the needed columns
        columns = self.get_columns()
        X_ready = X_clean[:, columns]

        # execute the cluster with the quantities of interest in dynamic
        # decomposition
        self.fit_transform(X_ready, y_clean)

        # retrieve and fix the labels
        labels = self.labels_
        self.labels_ = self.label_dirty(X, labels, clean_mask)

        # return the instance
        return self


# =============================================================================
# NEW!
# =============================================================================

import abc

import attr

import pandas as pd


_CIRCULARITY_ATTRIBUTES = ("normalized_star_energy", "eps", "eps_r")

_PTYPES_ORDER = tuple(p.name.lower() for p in data.ParticleSetType)


def hparam(default, **kwargs):
    metadata = kwargs.pop("metadata", {})
    metadata["__gchop_model_hparam__"] = True
    return attr.ib(default=default, metadata=metadata, **kwargs)


@attr.s(frozen=True, repr=False)
class GalaxyDecomposerABC(metaclass=abc.ABCMeta):

    bins = hparam(default=(0.05, 0.005))

    # block meta checks =======================================================
    def __init_subclass__(cls):
        model_config = getattr(cls, "__gchop_model_cls_config__", {})
        attr.s(maybe_cls=cls, **model_config)

    # block  to implement in every method =====================================

    @abc.abstractmethod
    def get_attributes(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def split(self, X, y, attributes):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_rows_mask(self, X, y, attributes):
        raise NotImplementedError()

    # internal ================================================================

    def __repr__(self):
        clsname = type(self).__name__

        selfd = attr.asdict(
            self,
            recurse=False,
            filter=lambda attr, _: attr.repr,
        )
        attrs_str = ",".join([f"{k}={v}" for k, v in selfd.items()])
        return f"{clsname}({attrs_str})"

    # API =====================================================================

    def _get_jcirc_df(self, galaxy, attributes):

        # STARS
        # turn the galaxy into jcirc dict
        # all the calculation cames together so we can't optimize here
        jcirc = utils.jcirc(galaxy, *self.bins)._asdict()

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
        gas_columns["ptypev"] = data.ParticleSetType.DARK_MATTER.value

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
        for attr in attributes:
            container = (
                circ_attrs if attr in _CIRCULARITY_ATTRIBUTES else df_attrs
            )
            container.append(attr)

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
        labels = self.split(X=X_clean, y=y_clean, attributes=attributes)

        # retrieve and fix the labels
        final_labels = self.complete_labels(
            X=X, labels=labels, rows_mask=rows_mask
        )

        # return the instance
        return final_labels, y


class DynamicStarsDecomposerMixin:
    def get_attributes(self):
        return ["normalized_star_energy", "eps", "eps_r"]

    def get_rows_mask(self, X, y, attributes):
        # all the rows where every value is finite
        only_stars = y == data.ParticleSetType.STARS.value
        finite_values = y == data.ParticleSetType.STARS.value
        return only_stars & finite_values
