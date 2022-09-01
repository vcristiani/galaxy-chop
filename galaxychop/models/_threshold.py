# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Monodimensional Models."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from ._base import DynamicStarsDecomposerMixin, GalaxyDecomposerABC, hparam
from ..preproc import doc_inherit


# =============================================================================
# Circularity Threshold
# =============================================================================


class JThreshold(DynamicStarsDecomposerMixin, GalaxyDecomposerABC):
    """JThreshold class.

    Implementation of galaxy dynamical decomposition model using only the
    circularity parameter. Tissera et al.(2012) [2]_,
    Marinacci et al.(2014) [3]_, Vogelsberger et al.(2014) [4]_,
    Park et al.(2019) [5]_ .

    Parameters
    ----------
    eps_cut : float, default=0.6
        Cut-off value in the circularity parameter. Stellar particles with
        eps > eps_cut are assigned to the disk and stellar particles with
        eps <= eps_cut to the spheroid.

    Notes
    -----
    Index of the cluster each stellar particles belongs to:
        Index=0: correspond to galaxy spheroid.
        Index=1: correspond to galaxy disk.

    Examples
    --------
    Example of implementation.

    >>> import galaxychop as gchop
    >>> galaxy = gchop.read_hdf5(...)
    >>> galaxy = gchop.utils.star_align(gchop.utils.center(galaxy))
    >>> chopper = gchop.JThreshold()
    >>> chopper.decompose(galaxy)

    References
    ----------
    .. [2] Tissera, P. B., White, S. D. M., and Scannapieco, C.,
        “Chemical signatures of formation processes in the stellar
        populations of simulated galaxies”,
        Monthly Notices of the Royal Astronomical Society, vol. 420, no. 1,
        pp. 255-270, 2012. doi:10.1111/j.1365-2966.2011.20028.x.
        `<https://ui.adsabs.harvard.edu/abs/2012MNRAS.420..255T/abstract>`_
    .. [3] Marinacci, F., Pakmor, R., and Springel, V.,
        “The formation of disc galaxies in high-resolution moving-mesh
        cosmological simulations”, Monthly Notices of the Royal Astronomical
        Society, vol. 437, no. 2, pp. 1750-1775, 2014.
        doi:10.1093/mnras/stt2003.
        `<https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1750M/abstract>`_
    .. [4] Vogelsberger, M., “Introducing the Illustris Project: simulating
        the coevolution of dark and visible matter in the Universe”,
        Monthly Notices of the Royal Astronomical Society, vol. 444, no. 2,
        pp. 1518-1547, 2014. doi:10.1093/mnras/stu1536.
        `<https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V/abstract>`_
    .. [5] Park, M.-J., “New Horizon: On the Origin of the Stellar Disk and
        Spheroid of Field Galaxies at z = 0.7”, The Astrophysical Journal,
        vol. 883, no. 1, 2019. doi:10.3847/1538-4357/ab3afe.
        `<https://ui.adsabs.harvard.edu/abs/2019ApJ...883...25P/abstract>`_
    """

    eps_cut = hparam(default=0.6)

    @eps_cut.validator
    def check_eps_cut(self, attribute, value):
        """Eps_cut value validator.

        This method validates that the value of eps_cut is in the interval
        (-1,1).
        """
        eps_cut = self.eps_cut
        if eps_cut > 1.0 or eps_cut < -1.0:
            raise ValueError(
                "The cut-off value in the circularity parameter is not between"
                f"(-1,1). Got eps_cut {eps_cut}"
            )

    @doc_inherit(GalaxyDecomposerABC.get_attributes)
    def get_attributes(self):
        """
        Notes
        -----
        In this model the parameter space is given by
            eps: circularity parameter (J_z/J_circ).
        """
        return ["eps"]

    @doc_inherit(GalaxyDecomposerABC.split)
    def split(self, X, y, attributes):
        """
        Notes
        -----
        The attributes used by the model are described in detail in the class
        documentation.
        """
        eps_cut = self.eps_cut

        esf_idx = np.where(X <= eps_cut)[0]
        disk_idx = np.where(X > eps_cut)[0]

        labels = np.empty(len(X), dtype=int)
        labels[esf_idx] = 0
        labels[disk_idx] = 1

        return labels, None

    @doc_inherit(GalaxyDecomposerABC.get_lmap)
    def get_lmap(self):
        return {0: "Spheroid", 1: "Disk"}
