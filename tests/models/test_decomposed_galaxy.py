# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

import galaxychop as gchop

import numpy as np

import pandas as pd

import pytest





# =============================================================================
# DECOMPOSEDGALAXY
# =============================================================================


@pytest.mark.model
def test_Decomposedgalaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    class Decomposer(gchop.models.GalaxyDecomposerABC):
        def get_stellar_attributes(self):
            return ["eps"]

        def split(self, X, y, stellar_properties):
            return np.full(len(X), 100), None

    decomposer = Decomposer()

    gal_decomp = decomposer.decompose(gal)

    assert len(gal_decomp) == len(gal)
    assert isinstance(gal_decomp, gchop.models.DecomposedGalaxy)
