# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""test for galaxychop.cli"""

# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop import cli, core, decomposers, io, preproc

import matplotlib.pyplot as plt

import pytest

from typer.testing import CliRunner


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def runner():
    return CliRunner()


# =============================================================================
# TESTS
# =============================================================================


def test_available_decomposers():
    available = cli.available_decomposers()

    assert set(available) == {
        "JThreshold",
        "JHistogram",
        "JEHistogram",
        "KMeans",
        "GaussianMixture",
        "AutoGaussianMixture",
    }
    assert available["JHistogram"] is decomposers.JHistogram


def test_methods(runner):
    result = runner.invoke(cli.app, ["methods"])

    assert result.exit_code == 0
    assert result.stdout.split() == list(cli.available_decomposers())


def test_info_galaxy(runner, data_path):
    path = data_path("gal394242.h5")

    result = runner.invoke(cli.app, ["info", str(path)])

    assert result.exit_code == 0
    expected = io.read_hdf5(path).total_mass().to_string()
    assert result.stdout.strip() == expected.strip()


def test_info_decomposed_galaxy(runner, data_path, tmp_path):
    output = tmp_path / "decomposed.h5"
    dgal = decomposers.JThreshold().decompose(
        preproc.center_and_align(
            io.read_hdf5(data_path("gal394242.h5")), r_cut=30
        )
    )
    io.to_hdf5(output, dgal)

    result = runner.invoke(cli.app, ["info", str(output)])

    assert result.exit_code == 0
    expected = io.read_hdf5(output).total_mass().to_string()
    assert result.stdout.strip() == expected.strip()
    assert "Spheroid" in result.stdout and "Disk" in result.stdout


def test_info_missing_file(runner, tmp_path):
    result = runner.invoke(cli.app, ["info", str(tmp_path / "nope.h5")])

    assert result.exit_code != 0


@pytest.mark.slow
def test_decompose(runner, data_path, tmp_path):
    output = tmp_path / "decomposed.h5"

    result = runner.invoke(
        cli.app,
        [
            "decompose",
            str(data_path("gal394242.h5")),
            str(output),
            "--method",
            "JThreshold",
            "--r-cut",
            "30",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "method='JThreshold'" in result.stdout

    dgal = io.read_hdf5(output)
    assert isinstance(dgal, decomposers.DecomposedGalaxy)
    assert dgal.method == "JThreshold"
    assert set(dgal.stars.labels) == {"Disk", "Spheroid", "stars"}


def test_decompose_invalid_method(runner, data_path, tmp_path):
    output = tmp_path / "decomposed.h5"

    result = runner.invoke(
        cli.app,
        [
            "decompose",
            str(data_path("gal394242.h5")),
            str(output),
            "--method",
            "Nope",
        ],
    )

    assert result.exit_code != 0
    assert "Nope" in result.output
    assert not output.exists()


def test_decompose_does_not_overwrite(runner, data_path, tmp_path):
    output = tmp_path / "decomposed.h5"
    output.write_bytes(b"previous content")

    result = runner.invoke(
        cli.app,
        ["decompose", str(data_path("gal394242.h5")), str(output)],
    )

    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert output.read_bytes() == b"previous content"


@pytest.mark.slow
def test_decompose_overwrite(runner, data_path, tmp_path):
    output = tmp_path / "decomposed.h5"
    output.write_bytes(b"previous content")

    result = runner.invoke(
        cli.app,
        [
            "decompose",
            str(data_path("gal394242.h5")),
            str(output),
            "--method",
            "JThreshold",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert io.read_hdf5(output).method == "JThreshold"


def test_available_plots():
    assert cli.available_plots() == [
        "hist",
        "hist2d",
        "kde",
        "kde2d",
        "rotation_curve",
        "sdyn_hist",
        "sdyn_hist2d",
        "sdyn_kde",
        "sdyn_kde2d",
    ]


@pytest.mark.plot
def test_plot_saves_figure(runner, data_path, tmp_path):
    output = tmp_path / "plot.png"

    result = runner.invoke(
        cli.app,
        [
            "plot",
            str(data_path("gal394242.h5")),
            "--kind",
            "hist2d",
            "--x",
            "x",
            "--y",
            "y",
            "--ptype",
            "stars",
            "--ptype",
            "gas",
            "-o",
            str(output),
        ],
    )
    plt.close("all")

    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG")


@pytest.mark.plot
def test_plot_passes_the_options(runner, data_path, monkeypatch):
    calls = []

    # same signature as the real hist2d
    def hist2d(self, x="x", *, y="z", ptypes=None, lmap=None, **kwargs):
        calls.append({"x": x, "y": y, "ptypes": ptypes, **kwargs})

    monkeypatch.setattr(core.plot.GalaxyPlotter, "hist2d", hist2d)
    monkeypatch.setattr(cli.plt, "show", lambda: calls.append("show"))

    result = runner.invoke(
        cli.app,
        [
            "plot",
            str(data_path("gal394242.h5")),
            "--x",
            "x",
            "--ptype",
            "stars",
        ],
    )

    assert result.exit_code == 0, result.output
    # options not given (y) keep the plot defaults; no -o shows it
    assert calls == [{"x": "x", "y": "z", "ptypes": ["stars"]}, "show"]


def test_plot_invalid_kind(runner, data_path):
    result = runner.invoke(
        cli.app, ["plot", str(data_path("gal394242.h5")), "-k", "nope"]
    )

    assert result.exit_code != 0
    assert "nope" in result.output


@pytest.mark.parametrize(
    "kind, option",
    [("hist", ["--y", "z"]), ("sdyn_hist", ["--ptype", "stars"])],
)
def test_plot_option_not_taken(runner, data_path, kind, option):
    result = runner.invoke(
        cli.app,
        ["plot", str(data_path("gal394242.h5")), "-k", kind, *option],
    )

    assert result.exit_code != 0
    assert "doesn't take it" in result.output


def test_main(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "app", lambda: called.append(True))

    cli.main()

    assert called == [True]
