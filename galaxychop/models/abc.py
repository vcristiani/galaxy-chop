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

from . import new_dgalaxy as dgalaxy
from .. import constants as consts
from .. import core
from ..core import sdynamics as sdyn
from ..preproc import is_centered, is_star_aligned

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
        # all the rows where every value is finite
        only_stars = np.equal(y, core.ParticleSetType.STARS.value)
        finite_values = np.isfinite(X).all(axis=1)
        return only_stars & finite_values

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

    def get_lmap(self):
        """Map the numeric labels of the components into a human readable \
        text."""
        return {}

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
        return f"<{clsname} {attrs_str}>"

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
        for attr_name in attributes:
            if attr_name not in _CIRCULARITY_ATTRIBUTES:
                raise ValueError(
                    f"Attribute {attr_name} is not a circularity attribute"
                )

        # If we have JCIRC attributes =========================================
        #     I'm going to need a lot of NANs that represent that gas and dm
        #     have no circularity.
        attributes = list(attributes) + ["ptypev"]
        df = self._get_jcirc_df(galaxy, attributes)

        # remove if ptypev is duplicated
        # df = df.loc[:, ~df.columns.duplicated()]

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
            return np.full((len(X), 1), np.nan)

        # the number of particles are incorrect so we simple remove the data
        probs_shape = list(np.shape(probs)[1:])

        # we need this many rows
        complete_shape = tuple([len(X)] + probs_shape)

        # now we create the container for the probabilities
        new_probs = np.full(complete_shape, np.nan)

        # and now we inject the probs in the correct order
        new_probs[rows_mask] = probs

        return new_probs

    def humanize_components(self, X, labels, probs, component_label_mapper):

        def mapper(component, ptypev):
            ptype = core.ParticleSetType.mktype(ptypev).humanize()
            return component_label_mapper.get(component, ptype)

        coso = np.column_stack(
            (X[:, -1], labels, probs)
        )  # coso y df deberias cambiar el nombre
        probs_columns = [f"prob_{i}" for i in range(probs.shape[1])]
        df = pd.DataFrame(
            coso, columns=["ptypev", "component"] + probs_columns
        )

        df["label"] = df.apply(
            lambda x: mapper(x["component"], x["ptypev"]), axis=1
        )

        return df

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
        component = self.complete_labels(
            X=X, labels=sorted(labels), rows_mask=rows_mask
        )
        probs = self.complete_probs(X=X, probs=probs, rows_mask=rows_mask)

        # this make a series for convenience
        component_labels = self.humanize_components(
            X=X,
            labels=component,
            probs=probs,
            component_label_mapper=self.get_lmap(),
        )

        import ipdb

        ipdb.set_trace()

        component_labels = self.get_lmap().copy()

        cls_name = type(self).__name__

        return dgalaxy.DecomposedGalaxy(
            stars=galaxy.stars,
            dark_matter=galaxy.dark_matter,
            gas=galaxy.gas,
            method=cls_name,
            component=component,
            component_labels=component_labels,
            probabilities=probs,
        )
