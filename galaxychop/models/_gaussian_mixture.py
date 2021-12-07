# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Gaussian Mixture Models."""


# =============================================================================
# IMPORTS
# =============================================================================

import joblib

import numpy as np

from sklearn import mixture

from ._base import DynamicStarsDecomposerMixin, GalaxyDecomposerABC, hparam
from ..utils import doc_inherit


# =============================================================================
# GAUSSIAN ABC
# =============================================================================
class DynamicStarsGaussianDecomposerABC(
    DynamicStarsDecomposerMixin, GalaxyDecomposerABC
):
    """Dynamic Stars Gaussian Decomposer Class.

    Parameters
    ----------
    covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default="full"
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    tol : float, default=0.001
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    reg_covar : float, default=1e-06
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    max_iter : float, default=100
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    n_init : int, default=10
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    init_params : {‘kmeans’, ‘random’}, default="kmeans"
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    weights_init : array-like of shape (n_components, ),  default=None
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    means_init : array-like of shape (n_components, n_features), default=None
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    precisions_init : array-like, default=None
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    random_state : int, default=None
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    warm_start : bool, default=False
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    verbose : int, default=0
        Parameter of :py:class:`GaussianMixture` class into `scikit-learn`
        library.
    verbose_interval : int, default=10
    """

    covariance_type = hparam(default="full")
    tol = hparam(default=0.001)
    reg_covar = hparam(default=1e-06)
    max_iter = hparam(default=100)
    n_init = hparam(default=10)
    init_params = hparam(default="kmeans")
    weights_init = hparam(default=None)
    means_init = hparam(default=None)
    precisions_init = hparam(default=None)
    random_state = hparam(default=None, converter=np.random.default_rng)
    warm_start = hparam(default=False)
    verbose = hparam(default=0)
    verbose_interval = hparam(default=10)

    @doc_inherit(GalaxyDecomposerABC.get_attributes)
    def get_attributes(self):
        """
        Notes
        -----
        In this model the parameter space is given by
            normalized_star_energy: normalized specific energy of the stars
            eps: circularity parameter (J_z/J_circ)
            eps_r: projected circularity parameter (J_p/J_circ).
        """
        return ["normalized_star_energy", "eps", "eps_r"]


# =============================================================================
# GMM
# =============================================================================
class GaussianMixture(DynamicStarsGaussianDecomposerABC):
    """GaussianMixture class.

    Implementation of the method for dynamically decomposing galaxies
    described by Obreja et al.(2018) [7]_ .

    Parameters
    ----------
    n_components: int, default=2
        The number of mixture components. Parameter of
        :py:class:`GaussianMixture` class into `scikit-learn` library.
    **kwargs: key, value mappings
        Other optional keyword arguments are passed through to
        :py:class:`GaussianMixture` class into `scikit-learn` library.

    Notes
    -----
    More information for `GaussianMixture` class:
        https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

    Examples
    --------
    Example of implementation of Gaussian Mixture Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.read_hdf5(...)
    >>> galaxy = gchop.star_align(gchop.center(galaxy))
    >>> chopper = gchop.GaussianMixture()
    >>> chopper.decompose(galaxy)

    References
    ----------
    .. [7] Obreja, A., “Introducing galactic structure finder: the multiple
        stellar kinematic structures of a simulated Milky Way mass galaxy”,
        Monthly Notices of the Royal Astronomical Society, vol. 477, no. 4,
        pp. 4915-4930, 2018. doi:10.1093/mnras/sty1022.
        `<https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.4915O/abstract>`_
        Obtain the columns of the quantities to be used.
    """

    n_components = hparam(default=2)

    @doc_inherit(GalaxyDecomposerABC.split)
    def split(self, X, y, attributes):
        """
        Notes
        -----
        The attributes used by the model are described in detail in the class
        documentation.
        """
        random_state = np.random.RandomState(self.random_state.bit_generator)

        gmm = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            weights_init=self.weights_init,
            means_init=self.means_init,
            precisions_init=self.precisions_init,
            random_state=random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )

        gmm_ = gmm.fit(X)
        labels = gmm_.predict(X)
        proba = gmm_.predict_proba(X)
        return labels, proba


# =============================================================================
# AUTO GMM
# =============================================================================


class AutoGaussianMixture(DynamicStarsGaussianDecomposerABC):
    """AutoGaussianMixture class.

    Implementation of the auto-gmm method for dynamically decomposing galaxies
    described by Du et al.(2019) [8]_ .

    Parameters
    ----------
    c_bic: float, default=0.1
        Cut value of the criteria for the automatic choice of the number of
        gaussians.
    n_jobs : int, default=None
    **kwargs: key, value mappings
        Other optional keyword arguments are passed through to
        :py:class:`GaussianMixture` class into `scikit-learn` library.

    Notes
    -----
    Index of the cluster each stellar particles belongs to:
        Index of the cluster each stellar particles belongs to.
        Index=0: correspond to galaxy stellar halo.
        Index=1: correspond to galaxy bulge.
        Index=2: correspond to galaxy cold disk.
        Index=3: correspond to galaxy warm disk.

    Examples
    --------
    Example of implementation of auto-gmm model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.read_hdf5(...)
    >>> galaxy = gchop.star_align(gchop.center(galaxy))
    >>> chopper = gchop.AutoGaussianMixture()
    >>> chopper.decompose(galaxy)

    References
    ----------
    .. [8] Du, M., “Identifying Kinematic Structures in Simulated Galaxies
        Using Unsupervised Machine Learning”, The Astrophysical Journal,
        vol. 884, no. 2, 2019. doi:10.3847/1538-4357/ab43cc.
        `<https://ui.adsabs.harvard.edu/abs/2019ApJ...884..129D/abstract>`_
    """

    c_bic = hparam(default=0.1)
    n_jobs = hparam(default=None)

    _COMPONENTS_TO_TRY = np.arange(2, 16)

    def _run_gmm(self, X, n_components, random_state):
        gmm = mixture.GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            weights_init=self.weights_init,
            means_init=self.means_init,
            precisions_init=self.precisions_init,
            random_state=random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )
        gmm.fit(X)
        return gmm

    def _try_components(self, X, n_components, random_state):
        gmm = self._run_gmm(X, n_components, random_state)
        return gmm.bic(X) / len(X)

    @doc_inherit(GalaxyDecomposerABC.split)
    def split(self, X, y, attributes):

        # for simplicity we conver the default_rng to a scikit-learn
        # compatible RandomState
        random_state = np.random.RandomState(self.random_state.bit_generator)

        # we copy self.components_to_try for simplicity
        ctt = self._COMPONENTS_TO_TRY

        # no we need multiple seed to creates a parrallel run of the GMM
        seeds = random_state.randint(np.iinfo(np.int32).max, size=len(ctt))

        with joblib.Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, prefer="processes"
        ) as P:
            # make the method delayed
            try_components = joblib.delayed(self._try_components)

            # excecute the trrys in parallel
            bic_med = P(
                try_components(X, n_components, seed)
                for n_components, seed in zip(ctt, seeds)
            )

        # convert bic_med to array
        bic_med = np.asarray(bic_med)

        # continue as normal
        bic_min = np.sum(bic_med[-5:]) / 5.0
        delta_bic_ = bic_med - bic_min

        # Criteria for the choice of the number of gaussians.
        mask = np.where(delta_bic_ <= self.c_bic)[0]

        # Number of components
        number_of_gaussians = np.min(ctt[mask])

        # Clustering with gaussian mixture and the parameters obtained.
        gcgmm_ = self._run_gmm(X, number_of_gaussians, random_state)

        n_components = gcgmm_.n_components
        center = gcgmm_.means_
        predict_proba = gcgmm_.predict_proba(X)

        # We add up the probabilities to obtain the classification of the
        # different particles.
        halo = np.zeros(len(X))
        bulge = np.zeros(len(X))
        cold_disk = np.zeros(len(X))
        warm_disk = np.zeros(len(X))

        for i in range(n_components):
            if center[i, 1] >= 0.85:
                cold_disk = cold_disk + predict_proba[:, i]
            if (center[i, 1] < 0.85) & (center[i, 1] >= 0.5):
                warm_disk = warm_disk + predict_proba[:, i]
            if (center[i, 1] < 0.5) & (center[i, 0] >= -0.75):
                halo = halo + predict_proba[:, i]
            if (center[i, 1] < 0.5) & (center[i, 0] < -0.75):
                bulge = bulge + predict_proba[:, i]

        probability = np.column_stack((halo, bulge, cold_disk, warm_disk))
        labels = probability.argmax(axis=1)

        return labels, probability
