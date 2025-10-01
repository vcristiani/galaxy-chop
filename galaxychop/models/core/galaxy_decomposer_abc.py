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

from . import decomposed_galaxy
from ... import constants as consts
from ... import core
from ...core import sdynamics as sdyn
from ...preproc import is_centered, is_star_aligned

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
    get_valid_stellar_mask and identify_galactic_components.

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

    def get_valid_stellar_mask(self, X, y, attributes):
        """
        Mask for valid stellar particles to operate clustering.

        This method gets the mask for valid stellar particles to
        operate clustering.

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
        valid_stellar_mask : nd.array(m_particles)
            Mask only with valid stellar particles to operate the clustering.

        """
        # all the rows where every value is finite
        only_stellar_particles = np.equal(y, core.ParticleSetType.STARS.value)
        finite_dynamics_values = np.isfinite(X).all(axis=1)
        return only_stellar_particles & finite_dynamics_values

    @abc.abstractmethod
    def split(self, X, y, attributes):
        """
        Identify galactic components through clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        component_labels : np.ndarray(m_particles)
            1D array with the index of the galactic components to
            which each particle belongs.
            m_particles is the total number of particles with valid
            values to operate the clustering.

        membership_probabilities : np.ndarray(m_particles) or None
            Probabilities of the particles to belong to each component, in case
            the dynamic decomposition model includes them. Otherwise it adopts
            the value None.

        """
        raise NotImplementedError()

    def get_component_name_mapping(self):
        """Map numeric component labels into physical component names."""
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

    def _extract_stellar_dynamics_dataframe(self, galaxy, attributes):
        # STARS
        # turn the galaxy into jcirc dict
        # all the calculation cames together so we can't optimize here
        stellar_dynamics_dict = galaxy.stellar_dynamics(
            bin0=self.cbins[0],
            bin1=self.cbins[1],
            reassign=self.reassign,
        ).to_dict()

        # we add the colum with the types, all the values from
        # stellar_dynamics_dict are stars
        stellar_dynamics_dict["ptypev"] = core.ParticleSetType.STARS.value
        stellar_dynamics_df = pd.DataFrame(
            {attr: stellar_dynamics_dict[attr] for attr in attributes}
        )

        # DARK_MATTER
        dark_matter_count = len(galaxy.dark_matter)
        dark_matter_nans = np.full(dark_matter_count, np.nan)

        dark_matter_columns = {
            attr: dark_matter_nans for attr in attributes
        }
        dark_matter_columns["ptypev"] = core.ParticleSetType.DARK_MATTER.value

        dark_matter_df = pd.DataFrame(dark_matter_columns)

        # GAS
        gas_count = len(galaxy.gas)
        gas_nans = np.full(gas_count, np.nan)

        gas_columns = {attr: gas_nans for attr in attributes}
        gas_columns["ptypev"] = core.ParticleSetType.GAS.value

        gas_df = pd.DataFrame(gas_columns)

        return pd.concat(
            [stellar_dynamics_df, dark_matter_df, gas_df], ignore_index=True
        )

    def extract_stellar_dynamics_matrix(self, galaxy, attributes):
        """
        Matrix of stellar dynamical properties.

        This method obtains the matrix with the stellar particles
        and dynamical properties necessary to operate
        the galactic component identification.

        Parameters
        ----------
        galaxy : ``Galaxy class`` object
            Instance of Galaxy class.
        attributes : keys of ``ParticleSet class`` parameters
            Stellar particle attributes used to operate the clustering.

        Returns
        -------
        X : np.ndarray(n_particles, attributes)
            2D array where each file it is a diferent particle and each column
            is a dynamical attribute of the particles.
            n_particles is the total number of particles.
        y : np.ndarray(n_particles)
            1D array where is identified the nature of each particle:
            0 = STARS, 1=DM, 2=Gas. n_particles is the total number of
            particles.

        """
        # first we split the attributes between the ones from circularity
        # and the ones from "galaxy.to_dataframe()"
        for stellar_attribute in attributes:
            if stellar_attribute not in _CIRCULARITY_ATTRIBUTES:
                raise ValueError(
                    f"Attribute {stellar_attribute} "
                    "is not a circularity attribute"
                )

        # If we have JCIRC attributes =========================================
        #     I'm going to need a lot of NANs that represent that gas and dm
        #     have no circularity.
        all_properties = list(attributes) + ["ptypev"]
        dynamics_dataframe = self._extract_stellar_dynamics_dataframe(
            galaxy, all_properties
        )

        # remove if ptypev is duplicated
        # dynamics_dataframe =
        # dynamics_dataframe.loc[:, ~dynamics_dataframe.columns.duplicated()]

        # separate matrix and particle types
        X = dynamics_dataframe[all_properties].to_numpy()
        y = dynamics_dataframe.ptypev.to_numpy()

        return X, y

    def assign_components_to_all_particles(
        self, X, galactic_components, valid_stellar_mask
    ):
        """
        Assign galactic components to all particles.

        This method assigns the galactic component labels
        obtained from clustering to the stellar particles
        used for this purpose. The rest are assigned as label=Nan.

        Parameters
        ----------
        X : np.ndarray(n_particles, attributes)
            2D array where each file it is a diferent particle and each column
            is a parameter of the particles.
            n_particles is the total number of particles.
        galactic_components: np.ndarray(m_particles)
            1D array with the index of the galactic components to
            which each stellar particle belongs.
            m_particles is the total number of particles with valid
            values to operate the clustering.
        valid_stellar_mask : nd.array(m_particles)
            Mask only with valid stellar particles to operate the clustering.
            m_particles is the total number of particles
            with valid values to operate the clustering.

        Return
        ------
        full_component_assignment: np.ndarray(n_particles)
            1D array with the index of the galactic components to which
            each particle belongs.
            Particles that do not belong to any of them are assigned
            the label Nan. n_particles is the total number of particles.
        """
        full_component_assignment = np.full(len(X), np.nan)
        full_component_assignment[valid_stellar_mask] = galactic_components
        return full_component_assignment

    def assign_probabilities_to_all_particles(
        self, X, membership_probabilities, valid_stellar_mask
    ):
        """
        Assign membership probabilities to all particles.

        This method assigns the membership probabilities
        obtained from clustering to the stellar particles used
        for this purpose, the rest are assigned as label=Nan.
        This method returns None in case the clustering method returns None
        probabilities.

        Parameters
        ----------
        X : np.ndarray(n_particles, attributes)
            2D array where each file it is a diferent particle and each column
            is a parameter of the particles.
            n_particles is the total number of particles.
        membership_probabilities: np.ndarray(n_cluster, m_particles)
            2D array with probabilities of belonging
            to each galactic component.
            n_cluster is the number of components obtained. m_particles is the
            total number of particles with valid values to operate the
            clustering.
        valid_stellar_mask : nd.array(m_particles)
            Mask only with valid stellar particles to
            operate the clustering.
            m_particles is the total number of particles
            with valid values to operate the clustering.

        Return
        ------
        full_membership_probabilities:
            np.ndarray(n_cluster, n_particles)
            2D array with probabilities of belonging
            to each galactic component.
            n_cluster is the number of components obtained. n_particles is the
            total number of particles. Particles that do not belong to any
            component are assigned the label Nan. This method returns None in
            case the clustering method returns None probabilities.

        """
        if membership_probabilities is None:
            return np.full((len(X), 1), np.nan)

        # the number of particles are incorrect so we simple remove the data
        prob_shape = list(np.shape(membership_probabilities)[1:])

        # we need this many rows
        complete_shape = tuple([len(X)] + prob_shape)

        # now we create the container for the probabilities
        full_membership_probabilities = np.full(complete_shape, np.nan)

        # and now we inject the probs in the correct order
        full_membership_probabilities[valid_stellar_mask] = (
            membership_probabilities
        )

        return full_membership_probabilities

    def create_physical_component_labels(
        self,
        X,
        full_component_assignment,
        full_membership_probabilities,
        component_name_mapper,
    ):
        """
        Create a DataFrame with component assignments and membership probs.

        This method takes the particle component assignments, the membership
        probabilities for each component, and a mapping of galactic component
        indices to human-readable names. It combines this information into
        a DataFrame, including a `label` column with a descriptive physical
        name for each particle.

        Parameters
        ----------
        X : np.ndarray
            Input matrix where the last column (`X[:, -1]`) contains
            the particle type values.
        full_component_assignment : np.ndarray
            Array with the component assignment for each particle.
        full_membership_probabilities : np.ndarray
            Matrix with membership probabilities for each component
            (rows = particles, columns = components).
        component_name_mapper : dict
            Dictionary mapping a galactic component index to a
            human-readable name.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the following columns:
            - `ptypev`: particle type value.
            - `component`: assigned component index.
            - `prob_i`: probability columns for each component.
            - `label`: descriptive physical name from the mapping.
        """

        def physical_name_mapper(galactic_component, particle_type_value):
            particle_type_name = core.ParticleSetType.mktype(
                particle_type_value
            ).humanize()
            return component_name_mapper.get(
                galactic_component, particle_type_name
            )

        particle_component_data = np.column_stack(
            (
                X[:, -1],
                full_component_assignment,
                full_membership_probabilities,
            )
        )
        probability_columns = [
            f"prob_{i}" for i in range(full_membership_probabilities.shape[1])
        ]
        component_dataframe = pd.DataFrame(
            particle_component_data,
            columns=["ptypev", "component"] + probability_columns,
        )

        component_dataframe["label"] = component_dataframe.apply(
            lambda x: physical_name_mapper(x["component"], x["ptypev"]), axis=1
        )

        return component_dataframe

    def _create_decomposed_particle_set(self, components_df, pset):
        data = components_df[components_df.ptypev == pset.ptype]

        prob_columns = data.columns[data.columns.str.startswith("prob_")]

        components = data.component.to_numpy(copy=True)
        labels = data.label.to_numpy(copy=True)
        probabilities = data[prob_columns].to_numpy(copy=True)

        component_pset = decomposed_galaxy.ComponentParticleSet.from_pset(
            pset,
            components=components,
            labels=labels,
            probabilities=probabilities,
        )

        return component_pset

    def decompose(self, galaxy):
        """
        Decompose galaxy into its structural components.

        Assign the galactic component (disk, bulge, stellar halo, etc.)
        to which each stellar particle belongs.
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
        # =====================================================================
        # 1. Galaxy preparation validation
        # =====================================================================
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

        # =====================================================================
        # 2. Extract stellar dynamical properties
        # =====================================================================
        attributes = self.get_attributes()

        X, y = self.extract_stellar_dynamics_matrix(
            galaxy, attributes=attributes
        )

        # =====================================================================
        # 3. Select valid stellar particles
        # =====================================================================
        valid_stellar_mask = self.get_valid_stellar_mask(
            X=X, y=y, attributes=attributes
        )
        X_clean, y_clean = X[valid_stellar_mask], y[valid_stellar_mask]

        # =====================================================================
        # 4. Identify galactic components
        # =====================================================================
        galactic_components, membership_probabilities = self.split(
            X=X_clean, y=y_clean, attributes=attributes
        )

        # =====================================================================
        # 5. Assign components to all particles
        # =====================================================================
        full_component_assignment = self.assign_components_to_all_particles(
            X=X,
            galactic_components=sorted(galactic_components),
            valid_stellar_mask=valid_stellar_mask,
        )
        full_membership_probabilities = (
            self.assign_probabilities_to_all_particles(
                X=X,
                membership_probabilities=membership_probabilities,
                valid_stellar_mask=valid_stellar_mask,
            )
        )

        # Convert component numbers to physical names (disk, bulge, halo, etc.)
        components_df = self.create_physical_component_labels(
            X=X,
            full_component_assignment=full_component_assignment,
            full_membership_probabilities=full_membership_probabilities,
            component_name_mapper=self.get_component_name_mapping(),
        )

        decomposition_method_name = type(self).__name__
        component_name_mapping = self.get_component_name_mapping().copy()

        stars_wc = self._create_decomposed_particle_set(
            components_df, galaxy.stars
        )
        dark_matter_wc = self._create_decomposed_particle_set(
            components_df, galaxy.dark_matter
        )
        gas_wc = self._create_decomposed_particle_set(
            components_df, galaxy.gas
        )

        del components_df

        # =====================================================================
        # 6. Build decomposed galaxy result
        # =====================================================================
        return decomposed_galaxy.DecomposedGalaxy(
            stars=stars_wc,
            dark_matter=dark_matter_wc,
            gas=gas_wc,
            method=decomposition_method_name,
            component_name_mapping=component_name_mapping,
        )
