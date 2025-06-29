# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""test for galaxy-chop.pipelines

"""


# =============================================================================
# IMPORTS
# =============================================================================


# import galaxychop as gchop

from galaxychop import pipeline
from galaxychop.models.gaussian_mixture import GaussianMixture
from galaxychop.models.threshold import JThreshold
from galaxychop.preproc.pcenter import Centralizer, center
from galaxychop.preproc.potential_energy import Potentializer
from galaxychop.preproc.salign import Aligner, star_align

import pandas as pd

import pytest


# =============================================================================
# TESTS
# =============================================================================


def test_pipeline_mkpipe(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")

    steps = [
        Centralizer(),
        Aligner(r_cut=30),
        JThreshold(),
    ]

    pipe = pipeline.mkpipe(*steps)
    result = pipe.decompose(gal)
    df_result = result.components.to_dataframe()

    gal_center = center(gal)
    gal_center_align = star_align(gal_center, r_cut=30)
    result_decompose_with_func = JThreshold().decompose(gal_center_align)
    df_result_decompose_with_func = (
        result_decompose_with_func.components.to_dataframe()
    )

    pd.testing.assert_frame_equal(df_result_decompose_with_func, df_result)


def test_pipeline_slicing():
    steps = [
        Centralizer(),
        Aligner(),
        Potentializer(),
        JThreshold(),
    ]

    pipe = pipeline.mkpipe(*steps)

    for idx, step in enumerate(steps):
        assert pipe[idx] == step

    for name, step in pipe.named_steps.items():
        assert pipe[name] == step

    assert [s for _, s in pipe[2:].steps] == steps[2:]

    with pytest.raises(ValueError):
        pipe[::2]

    with pytest.raises(KeyError):
        pipe[None]


def test_pipeline_not_transformer_fail():
    steps = [JThreshold(), GaussianMixture()]
    with pytest.raises(TypeError):
        pipeline.mkpipe(*steps)


def test_pipeline_not_decomposer_fail():
    steps = [Centralizer(), Aligner()]
    with pytest.raises(TypeError):
        pipeline.mkpipe(*steps)


def test_pipeline_name_not_str():
    with pytest.raises(TypeError):
        pipeline.GchopPipeline(
            steps=[(..., Centralizer()), ("final", JThreshold())]
        )
    with pytest.raises(TypeError):
        pipeline.GchopPipeline(
            steps=[("first", Centralizer()), (..., JThreshold())]
        )
