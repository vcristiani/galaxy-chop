# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
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


# =============================================================================
# GAUSSIAN ABC
# =============================================================================
class DynamicStarsGaussianDecomposerABC(
    DynamicStarsDecomposerMixin, GalaxyDecomposerABC
):
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

    def get_attributes(self):
        return ["normalized_star_energy", "eps", "eps_r"]


# =============================================================================
# GMM
# =============================================================================
class GaussianMixture(DynamicStarsGaussianDecomposerABC):
    """GalaxyChop Gaussian Mixture Model class.

    Implementation of the method for dynamically decomposing galaxies
    described by Obreja et al.(2018) [7]_ .

    Parameters
    ----------
    columns: default=None
        Physical quantities of stellars particles
        used to decompose galaxies.

    **kwargs: key, value mappings
        Other optional keyword arguments are passed through to
        :py:class:`GalaxyDecomposeMixin` and :py:class:`GaussianMixture`
        classes.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
        Index of the cluster each stellar particles belongs to.
    weights_ :
        Original attribute create by the `GaussianMixture`
        class into `scikit-learn` library.
    means_ :
        Original attribute create by the `GaussianMixture` class into
        `scikit-learn` library.
    covariances_ :
        Original attribute create by the `GaussianMixture`
        class into `scikit-learn` library.
    precisions_ :
        Original attribute create by the `GaussianMixture` class into
        `scikit-learn` library.
    precisions_cholesky_ :
        Original attribute create by the `GaussianMixture`
        class into `scikit-learn` library.
    converged_ :
        Original attribute create by the `GaussianMixture` class into
        `scikit-learn` library.
    n_iter_ :
        Original attribute create by the `GaussianMixture` class into
        `scikit-learn` library.
    lower_bound_ :
        Original attribute create by the `GaussianMixture`
        class into `scikit-learn` library.

    Notes
    -----
    n_components : int, default=1
        The number of mixture components. Parameter of
        :py:class:`GaussianMixture` class.
    More information for `GaussianMixture` class:
        https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

    Examples
    --------
    Example of implementation of CGGmm Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> chopper = gchop.GaussianMixture(n_components=3)
    >>> chopper.decompose(galaxy)
    >>> chopper.labels_
    array([-1, -1,  2, ...,  1,  2,  1])

    References
    ----------
    .. [7] Obreja, A., “Introducing galactic structure finder: the multiple
        stellar kinematic structures of a simulated Milky Way mass galaxy”,
        Monthly Notices of the Royal Astronomical Society, vol. 477, no. 4,
        pp. 4915-4930, 2018. doi:10.1093/mnras/sty1022.
        `<https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.4915O/abstract>`_
        Obtain the columns of the quantities to be used.

    Returns
    -------
    columns: list
        Only the needed columns used to decompose galaxies.
    """

    n_components = hparam(default=2)

    def split(self, X, y, attributes):
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
    """GalaxyChop auto-gmm class.

    Implementation of the method for dynamically decomposing galaxies
    described by Du et al.(2019) [8]_ .

    Parameters
    ----------
    c_bic: float, default=0.1
        Cut value of the criteria for the automatic choice of
        the number of gaussians.

    Attributes
    ----------
    labels_: `np.ndarray(n)`, n: number of particles with E<=0 and -1<eps<1.
        Index of the cluster each stellar particles belongs to.
        Index=0: correspond to galaxy stellar halo.
        Index=1: correspond to galaxy bulge.
        Index=2: correspond to galaxy cold disk.
        Index=3: correspond to galaxy warm disk.

    probability: np.ndarray(n,4), n:number of particles with E<=0 and -1<eps<1.
        Probability of each stellar particle to belong to each
        component of the galaxy.

    probability_of_gaussianmixturey: `np.ndarray(n_particles, n_gaussians)`.
        Probability of each stellar particle (with E<=0 and -1<eps<1) to belong
        to each gaussian.

    bic_med_: np.ndarray(number of component to try,1).
        BIC parameter(len(X)).
    gausians_: tuple(n_gausians).
        Number of gaussians used to choise the number of clusters.
    bic_min_: float.
        Mean value of BIC(n_c>10).
    delta_bic_: np.ndarray(number of component to try,1)
        `bic_med_` - `bic_min_`.
    mask_:  np.ndarray(number of gaussians).
        Index of components to try that fulfil with c_BIC criteria.
    n_components_: int.
        Number of gaussians automatically selected.
    gcgmm_: object.
        Original `gcgmm_` object create by the `GaussianMixture` class
        with number of cluster automatically selected.

    Examples
    --------
    Example of implementation of CGAutogmm Model.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.Galaxy(...)
    >>> chopper = gchop.AutoGaussianMixture(c_bic=0.1)
    >>> chopper.decompose(galaxy)
    >>> chopper.labels_
    array([-1, -1,  1, ...  0, 0, 3])

    References
    ----------
    .. [8] Du, M., “Identifying Kinematic Structures in Simulated Galaxies
        Using Unsupervised Machine Learning”, The Astrophysical Journal,
        vol. 884, no. 2, 2019. doi:10.3847/1538-4357/ab43cc.
        `<https://ui.adsabs.harvard.edu/abs/2019ApJ...884..129D/abstract>`_
    """

    c_bic = hparam(default=0.1)
    n_jobs = hparam(default=None)

    _COMPONENTS_TO_TRY = tuple(range(2, 16))

    def _try_components(self, X, n_components, seed):
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
            random_state=seed,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )
        gmm.fit(X)
        return gmm.bic(X) / len(X)

    def split(self, X, y, attributes):

        # for simplicity we conver the default_rng to a scikit-learn
        # compatible RandomState
        random_state = np.random.RandomState(self.random_state.bit_generator)

        # we copy self.components_to_try for simplicity
        ctt = np.asarray(self._COMPONENTS_TO_TRY)

        # no we need multiple seed to creates a parrallel run of the GMM
        seeds = random_state.randint(np.iinfo(np.int32).max, size=len(ctt))

        with joblib.Parallel(
            n_jobs=8, verbose=self.verbose, prefer="processes"
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
        gcgmm = mixture.GaussianMixture(
            n_components=number_of_gaussians,
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

        gcgmm_ = gcgmm.fit(X)

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
