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

import abc
import warnings

import attr

import numpy as np

import pandas as pd

from .. import constants as consts
from .. import core
from ..core import sdynamics as sdyn
from ..preproc import is_centered, is_star_aligned
from ..utils import doc_inherit
from . import dgalaxy

# =============================================================================
# CONSTANTS
# =============================================================================

_CIRCULARITY_ATTRIBUTES = sdyn._GalaxyStellarDynamics.circularity_attributes()

_PTYPES_ORDER = tuple(p.name.lower() for p in core.ParticleSetType)


# =============================================================================
# FUNCTIONS
# =============================================================================

def hparam(default, **kwargs):
    """
    Create a hyper parameter for decomposers.

    By design decision, hyper-parameter is required to have a sensitive default
    value.

    Parameters
    ----------
    default :
        Sensitive default value of the hyper-parameter.
    **kwargs :
        Additional keyword arguments are passed and are documented in
        ``attr.ib()``.

    Return
    ------
    Hyper parameter with a default value.

    Notes
    -----
    This function is a thin-wrapper over the attrs function ``attr.ib()``.
    """
    metadata = kwargs.pop("metadata", {})
    metadata["__gchop_model_hparam__"] = True
    return attr.ib(default=default, metadata=metadata, kw_only=True, **kwargs)


# =============================================================================
# ABC
# =============================================================================


@attr.s(frozen=True, repr=False)
class GalaxyDecomposerABC(metaclass=abc.ABCMeta):
    """
    Abstract class to facilitate the creation of decomposers.

    This class requests the redefinition of three methods: get_attributes,
    get_rows_mask and split.

    Parameters
    ----------
    cbins : tuple
        It contains the two widths of bins necessary for the calculation of the
        circular angular momentum.  Shape: (2,). Dafult value = (0.05, 0.005).
    reassign : list
        It allows to define what to do with stellar particles with circularity
        parameter values >1 or <-1. Default value = [False].

    """

    __gchop_model_cls_config__ = {"repr": False, "frozen": True}

    cbins = hparam(default=consts.SD_DEFAULT_CBIN)

    @cbins.validator
    def _bins_validator(self, attribute, value):
        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], float)
            and isinstance(value[1], float)
        ):
            raise ValueError("cbins must be a tuple of two floats.")

    reassign = hparam(
        default=consts.SD_DEFAULT_REASSIGN,
        validator=attr.validators.instance_of(bool),
    )

    # block meta checks =======================================================
    def __init_subclass__(cls):
        """
        Initiate of subclasses.

        It ensures that every inherited class is decorated by ``attr.s()`` and
        assigns as class configuration the parameters defined in the class
        variable `__gchop_model_cls_config__`.

        In other words it is slightly equivalent to:

        .. code-block:: python

            @attr.s(**GalaxyDecomposerABC.__gchop_model_cls_config__)
            class Decomposer(GalaxyDecomposerABC):
                pass

        """
        model_config = getattr(cls, "__gchop_model_cls_config__")
        attr.s(maybe_cls=cls, **model_config)

    # block  to implement in every method =====================================

    @abc.abstractmethod
    def get_attributes(self):
        """
        Attributes for the parameter space.

        Returns
        -------
        attributes : keys of ``ParticleSet class`` parameters
            Particle attributes used to operate the clustering.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_rows_mask(self, X, y, attributes):
        """
        Mask for the valid rows to operate clustering.

        This method gets the mask for the valid rows to operate clustering.

        Parameters
        ----------
        X : np.ndarray(n_particles, attributes)
            2D array where each file it is a diferent particle and each column
            is an attribute of the particles.
            n_particles is the total number of particles.
        y : np.ndarray(n_particles,)
            1D array where is identified the type of each particle:
            0 = stars, 1 = dark matter, 2 = gas. n_particles is the total
            number of particles.
        attributes : tuple
            Dictionary keys of ``ParticleSet class`` parameters with particle
            attributes used to operate the clustering.

        Returns
        -------
        mask : nd.array(m_particles)
            Mask only with valid values to operate the clustering.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def split(self, X, y, attributes):
        """
        Compute clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : np.ndarray(m_particles)
            1D array with the index of the clusters to which each particle
            belongs. m_particles is the total number of particles with valid
            values to operate the clustering.

        probs : np.ndarray(m_particles) or None
            Probabilities of the particles to belong to each component, in case
            the dynamic decomposition model includes them. Otherwise it adopts
            the value None.

        """
        raise NotImplementedError()

    # internal ================================================================

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
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
        jcirc = galaxy.stellar_dynamics(
            bin0=self.cbins[0],
            bin1=self.cbins[1],
            reassign=self.reassign,
        ).to_dict()

        # we add the colum with the types, all the values from jcirc
        # are stars
        jcirc["ptypev"] = core.ParticleSetType.STARS.value
        stars_df = pd.DataFrame({attr: jcirc[attr] for attr in attributes})

        # DARK_MATTER
        dm_rows = len(galaxy.dark_matter)
        dm_nans = np.full(dm_rows, np.nan)

        dm_columns = {attr: dm_nans for attr in attributes}
        dm_columns["ptypev"] = core.ParticleSetType.DARK_MATTER.value

        dm_df = pd.DataFrame(dm_columns)

        # GAS
        gas_rows = len(galaxy.gas)
        gas_nans = np.full(gas_rows, np.nan)

        gas_columns = {attr: gas_nans for attr in attributes}
        gas_columns["ptypev"] = core.ParticleSetType.GAS.value

        gas_df = pd.DataFrame(gas_columns)

        return pd.concat([stars_df, dm_df, gas_df], ignore_index=True)

    def attributes_matrix(self, galaxy, attributes):
        """
        Matrix of particle attributes.

        This method obtains the matrix with the particles and attributes
        necessary to operate the clustering.

        Parameters
        ----------
        galaxy : ``Galaxy class`` object
            Instance of Galaxy class.
        attributes : keys of ``ParticleSet class`` parameters
            Particle attributes used to operate the clustering.

        Returns
        -------
        X : np.ndarray(n_particles, attributes)
            2D array where each file it is a diferent particle and each column
            is a attribute of the particles. n_particles is the total number of
            particles.
        y : np.ndarray(n_particles)
            1D array where is identified the nature of each particle:
            0 = STARS, 1=DM, 2=Gas. n_particles is the total number of
            particles.

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

        # separate matrix and classes
        X = df[attributes].to_numpy()
        y = df.ptypev.to_numpy()

        return X, y

    def complete_labels(self, X, labels, rows_mask):
        """
        Complete the labels of all particles.

        This method assigns the labels obtained from clustering to the
        particles used for this purpose. The rest are assigned as label=Nan.

        Parameters
        ----------
        X : np.ndarray(n_particles, attributes)
            2D array where each file it is a diferent particle and each column
            is a parameter of the particles. n_particles is the total number of
            particles.
        labels: np.ndarray(m_particles)
            1D array with the index of the clusters to which each particle
            belongs. m_particles is the total number of particles with valid
            values to operate the clustering.
        rows_mask : nd.array(m_particles)
            Mask only with valid values to operate the clustering. m_particles
            is the total number of particles with valid values to operate the
            clustering.

        Return
        ------
        new_labels: np.ndarray(n_particles)
            1D array with the index of the clusters to which each particle
            belongs. Particles that do not belong to any of them are assigned
            the label Nan. n_particles is the total number of particles.
        """
        new_labels = np.full(len(X), np.nan)
        new_labels[rows_mask] = labels
        return new_labels

    def complete_probs(self, X, probs, rows_mask):
        """
        Complete the probabilities of all particles.

        This method assigns the probabilities obtained from clustering to the
        particles used for this purpose, the rest are assigned as label=Nan.
        This method returns None in case the clustering method returns None
        probabilities.

        Parameters
        ----------
        X : np.ndarray(n_particles, attributes)
            2D array where each file it is a diferent particle and each column
            is a parameter of the particles. n_particles is the total number of
            particles.
        probs: np.ndarray(n_cluster, m_particles)
            2D array with probabilities of belonging to each component.
            n_cluster is the number of components obtained. m_particles is the
            total number of particles with valid values to operate the
            clustering.
        rows_mask : nd.array(m_particles)
            Mask only with valid values to operate the clustering. m_particles
            is the total number of particles with valid values to operate the
            clustering.

        Return
        ------
        new_probs: np.ndarray(n_cluster, n_particles)
            2D array with probabilities of belonging to each component.
            n_cluster is the number of components obtained. n_particles is the
            total number of particles. Particles that do not belong to any
            component are assigned the label Nan. This method returns None in
            case the clustering method returns None probabilities.

        """
        if probs is None:
            return None

        # the number of particles are incorrect so we simple remove the data
        probs_shape = list(np.shape(probs)[1:])

        # we need this many rows
        complete_shape = tuple([len(X)] + probs_shape)

        # now we create the container for the probabilities
        new_probs = np.full(complete_shape, np.nan)

        # and now we inject the probs in the correct order
        new_probs[rows_mask] = probs

        return new_probs

    def get_lmap(self):
        """Map the numeric labels of the components into a human readable \
        text."""
        return {}

    def decompose(self, galaxy):
        """
        Decompose method.

        Assign the component of the galaxy to which each particle belongs.
        Validation of the input galaxy instance.

        Parameters
        ----------
        galaxy : ``Galaxy class`` object
            Instance of Galaxy class.

        Return
        ------
        Components :
            Instance of the ``Component class``, with the result of the dynamic
            decomposition.

        """
        # Before anything, check if centered and aligned (!)
        if not is_centered(galaxy):
            warnings.warn(
                "Input Galaxy is not centered. Please, center it \
                    with Centralizer.transform(galaxy) \
                    or proceed with caution.",
                UserWarning,
            )

        if not is_star_aligned(galaxy):
            warnings.warn(
                "Input Galaxy is not aligned. Please, align it \
                    with Aligner.transform(galaxy) \
                    or proceed with caution.",
                UserWarning,
            )

        attributes = self.get_attributes()

        X, y = self.attributes_matrix(galaxy, attributes=attributes)

        # calculate only the valid values to operate the clustering
        rows_mask = self.get_rows_mask(X=X, y=y, attributes=attributes)
        X_clean, y_clean = X[rows_mask], y[rows_mask]

        # execute the cluster with the quantities of interest
        labels, probs = self.split(X=X_clean, y=y_clean, attributes=attributes)

        # retrieve and fix the labels
        final_labels = self.complete_labels(
            X=X, labels=sorted(labels), rows_mask=rows_mask
        )
        final_probs = self.complete_probs(
            X=X, probs=probs, rows_mask=rows_mask
        )
        final_y = np.array(
            [core.ParticleSetType.mktype(yi).humanize() for yi in y]
        )

        # return the instance
        mass = galaxy.to_dataframe(
            ptypes=_PTYPES_ORDER, attributes=["m"]
        ).m.to_numpy()

        # we make the components and wrap they with the galaxy
        # in a "DecomposedGalaxy" class.
        components = dgalaxy.Components(
            labels=final_labels,
            ptypes=final_y,
            probabilities=final_probs,
            m=mass,
            lmap=self.get_lmap().copy(),
        )

        return dgalaxy.DecomposedGalaxy(galaxy, components)


# =============================================================================
# MIXIN
# =============================================================================


class DynamicStarsDecomposerMixin:
    """
    Dynamic Stars Decomposer Mixin Class.

    This class redefines the get_row_mask method so that dynamic decomposition
    is performed using only stellar particles.

    """

    @doc_inherit(GalaxyDecomposerABC.get_rows_mask)
    def get_rows_mask(self, X, y, attributes):
        """
        Note
        ----
        Only stellar particles are used to carry out the dynamic decomposition.
        In addition, the parameters of the parameter space, where the dynamic
        decomposition is carried out, must have finite values.
        """
        # all the rows where every value is finite
        only_stars = np.equal(y, core.ParticleSetType.STARS.value)
        finite_values = np.isfinite(X).all(axis=1)
        return only_stars & finite_values
