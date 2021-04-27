# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Gaussian Mixture Models."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from sklearn import mixture
from sklearn.base import ClusterMixin, TransformerMixin

from ._base import GalaxyDecomposeMixin

# =============================================================================
# GMM
# =============================================================================


class GaussianMixture(GalaxyDecomposeMixin, mixture.GaussianMixture):
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
    """

    def __init__(self, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns

    def get_columns(self):
        """Obtain the columns of the quantities to be used.

        Returns
        -------
        columns: list
            Only the needed columns used to decompose galaxies.
        """
        if self.columns is None:
            return super().get_columns()
        return self.columns

    def fit_transform(self, X, y=None):
        """Transform method.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed.
        """
        labels = self.fit(X).predict(X)
        self.labels_ = labels
        return self


# =============================================================================
# AUTO GMM
# =============================================================================


class AutoGaussianMixture(
    GalaxyDecomposeMixin, ClusterMixin, TransformerMixin
):
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

    def __init__(self, c_bic=0.1):
        self.c_bic = c_bic
        self.component_to_try = np.arange(2, 16)
        # self.component_to_try = (
        #     np.arange(2, 16) if component_to_try is None else
        # component_to_try
        # )

    def fit(self, X, y=None):
        """Compute AutoGaussianMixture clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        bic_med = np.empty(len(self.component_to_try))
        gausians = []
        for i in self.component_to_try:
            # Implementation of gmm for all possible components of the method.
            gmm = mixture.GaussianMixture(
                n_components=i, n_init=10, random_state=0
            )
            gmm.fit(X)
            bic_med[i - 2] = gmm.bic(X) / len(X)
            gausians.append(gmm)

        bic_min = np.sum(bic_med[-5:]) / 5.0
        delta_bic_ = bic_med - bic_min

        # Criteria for the choice of the number of gaussians.
        c_bic = self.c_bic
        mask = np.where(delta_bic_ <= c_bic)[0]

        # Number of components
        number_of_gaussians = np.min(self.component_to_try[mask])

        # Clustering with gaussian mixture and the parameters obtained.
        gcgmm_ = mixture.GaussianMixture(
            n_components=number_of_gaussians,
            random_state=0,
        )
        gcgmm_.fit(X)

        # store all in the instances
        self.bic_med_ = bic_med
        self.gausians_ = tuple(gausians)
        self.bic_min_ = bic_min
        self.delta_bic_ = delta_bic_
        self.c_bic_ = c_bic
        self.mask_ = mask
        self.n_components_ = number_of_gaussians
        self.gcgmm_ = gcgmm_

        return self

    def transform(self, X, y=None):
        """Transform method.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed.
        """
        n_components = self.n_components_
        center = self.gcgmm_.means_
        predict_proba = self.gcgmm_.predict_proba(X)

        # We add up the probabilities to obtain the classification of the
        # different particles.
        halo = np.zeros(len(X))
        bulge = np.zeros(len(X))
        cold_disk = np.zeros(len(X))
        warm_disk = np.zeros(len(X))

        for i in range(0, n_components):
            if center[i, 1] >= 0.85:
                cold_disk = cold_disk + predict_proba[:, i]
            if (center[i, 1] < 0.85) & (center[i, 1] >= 0.5):
                warm_disk = warm_disk + predict_proba[:, i]
            if (center[i, 1] < 0.5) & (center[i, 0] >= -0.75):
                halo = halo + predict_proba[:, i]
            if (center[i, 1] < 0.5) & (center[i, 0] < -0.75):
                bulge = bulge + predict_proba[:, i]

        probability = np.column_stack((halo, bulge, cold_disk, warm_disk))
        labels = np.empty(len(X), dtype=int)

        for i in range(len(X)):
            labels[i] = probability[i, :].argmax()

        self.probability_of_gaussianmixture = predict_proba
        self.probability = probability
        self.labels_ = labels

        return self
