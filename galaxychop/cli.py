# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Command line interface of GalaxyChop.

Installed as the ``galaxychop`` command::

    $ galaxychop methods
    $ galaxychop info decomposed.h5
    $ galaxychop plot decomposed.h5 --kind sdyn_hist2d -o plot.png
    $ galaxychop decompose galaxy.h5 decomposed.h5 --method JHistogram

"""

# =============================================================================
# IMPORTS
# =============================================================================

import inspect
import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt

import typer

from . import core, decomposers, io, preproc

# =============================================================================
# APP
# =============================================================================

app = typer.Typer(
    help="GalaxyChop: dynamical decomposition of galaxies.",
    no_args_is_help=True,
    add_completion=False,
)


def available_decomposers():
    """
    Return the decomposers that can be used from the command line.

    Returns
    -------
    dict
        Maps each concrete decomposer class name (e.g. ``"JHistogram"``)
        in ``galaxychop.decomposers`` to its class.
    """
    found = {}
    for name in decomposers.__all__:
        obj = getattr(decomposers, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, decomposers.GalaxyDecomposerABC)
            and not inspect.isabstract(obj)
        ):
            found[name] = obj
    return found


def available_plots():
    """
    Return the plots that can be drawn from the command line.

    Returns
    -------
    list of str
        Names of the public plot methods of ``GalaxyPlotter`` (e.g.
        ``"hist2d"``), the same ones ``galaxy.plot(plot_kind)`` accepts.
    """
    plotter = core.plot.GalaxyPlotter
    forbidden = plotter.P_KIND_FORBIDDEN_METHODS
    return [
        name
        for name, _ in inspect.getmembers(plotter, inspect.isfunction)
        if not name.startswith("_") and name not in forbidden
    ]


# =============================================================================
# COMMANDS
# =============================================================================


@app.command()
def methods():
    """List the available decomposition methods."""
    for name in available_decomposers():
        typer.echo(name)


@app.command()
def info(
    path: pathlib.Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="HDF5 file with a galaxy or a decomposed galaxy.",
    ),
):
    """Print the total mass of a galaxy or of each of its components."""
    galaxy = io.read_hdf5(path)
    typer.echo(galaxy.total_mass().to_string())


@app.command()
def plot(
    path: pathlib.Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="HDF5 file with a galaxy or a decomposed galaxy.",
    ),
    kind: str = typer.Option(
        "hist2d",
        "--kind",
        "-k",
        help="Plot to draw: one of the galaxy.plot methods (e.g. hist2d, "
        "kde2d, rotation_curve, sdyn_hist2d).",
    ),
    x: Optional[str] = typer.Option(None, help="Attribute on the x axis."),
    y: Optional[str] = typer.Option(None, help="Attribute on the y axis."),
    ptypes: Optional[List[str]] = typer.Option(
        None,
        "--ptype",
        help="Particle type to plot (stars, dark_matter or gas); repeat "
        "it for several.",
    ),
    output: Optional[pathlib.Path] = typer.Option(
        None,
        "--output",
        "-o",
        dir_okay=False,
        help="Save the figure in this file instead of showing it.",
    ),
):
    """Plot a galaxy (by particle type) or a decomposed one (by component)."""
    available = available_plots()
    if kind not in available:
        raise typer.BadParameter(
            f"{kind!r}. Choose one of: {', '.join(available)}",
            param_hint="--kind",
        )

    galaxy = io.read_hdf5(path)
    method = getattr(galaxy.plot, kind)

    # only pass the options this plot takes, so an unsupported one is a
    # clear error here instead of a seaborn error about its **kwargs
    params = inspect.signature(method).parameters
    options = {"x": ("--x", x), "y": ("--y", y), "ptypes": ("--ptype", ptypes)}
    kwargs = {}
    for name, (flag, value) in options.items():
        if not value:
            continue
        if name not in params:
            raise typer.BadParameter(
                f"the {kind!r} plot doesn't take it", param_hint=flag
            )
        kwargs[name] = value

    method(**kwargs)

    if output is None:
        plt.show()
    else:
        plt.gcf().savefig(output, bbox_inches="tight")
        typer.echo(f"Saved {output}")


@app.command()
def decompose(
    path: pathlib.Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="HDF5 file with a galaxy."
    ),
    output: pathlib.Path = typer.Argument(
        ..., dir_okay=False, help="HDF5 file to store the decomposition."
    ),
    method: str = typer.Option(
        "JHistogram",
        "--method",
        "-m",
        help="Decomposition method (see `galaxychop methods`).",
    ),
    align: bool = typer.Option(
        True,
        help="Center and align the galaxy before decomposing it.",
    ),
    r_cut: Optional[float] = typer.Option(
        None,
        help="Radius (kpc) of the stars used to align the galaxy.",
    ),
    overwrite: bool = typer.Option(
        False, help="Replace the output file if it already exists."
    ),
):
    """Decompose a galaxy into its components and store the result."""
    available = available_decomposers()
    if method not in available:
        raise typer.BadParameter(
            f"{method!r}. Choose one of: {', '.join(available)}",
            param_hint="--method",
        )

    # to_hdf5 appends to an existing file, so never mix with an old one
    if output.exists():
        if not overwrite:
            raise typer.BadParameter(
                f"{output} already exists (use --overwrite to replace it)",
                param_hint="OUTPUT",
            )
        output.unlink()

    galaxy = io.read_hdf5(path)
    if align:
        galaxy = preproc.center_and_align(galaxy, r_cut=r_cut)

    decomposed = available[method]().decompose(galaxy)
    io.to_hdf5(output, decomposed)

    typer.echo(repr(decomposed))
    typer.echo(decomposed.total_mass().to_string())


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Run the ``galaxychop`` command (entry point in pyproject.toml)."""
    app()
