# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2020, 2021, 2022, 2023, 2024, 2025, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Projected (IFU-like) maps of a galaxy."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

#: Supported projections: name -> (horizontal axis, vertical axis, line of
#: sight velocity). They assume the galaxy is aligned, i.e. the stellar
#: angular momentum points along z (see ``galaxychop.preproc.star_align``).
PROJECTIONS = {
    "face-on": ("x", "y", "vz"),
    "edge-on": ("x", "z", "vy"),
}

#: Percentile of the projected distance to the origin used as the default
#: half width of the map, so a few far-out particles don't blow it up.
DEFAULT_HALF_WIDTH_PCT = 99

# =============================================================================
# API
# =============================================================================


def projected_maps(
    galaxy,
    *,
    projection="edge-on",
    ptypes=("stars",),
    bins=50,
    half_width=None,
    min_particles=1,
    components=None,
):
    """
    Project the galaxy onto a square grid of pixels.

    Every pixel condenses the particles that fall on it (along the whole
    line of sight) into ``n_particles``, ``mass`` (M_sun),
    ``surface_density`` (M_sun / kpc**2) and the mass weighted line of sight
    velocity ``v_los`` and dispersion ``sigma_los`` (km/s), as an integral
    field spectrograph would see them.

    The grid is always complete and centered on the origin: empty pixels are
    kept, with zero mass and ``NaN`` velocities, so for the same ``bins``
    every map has the same shape.

    Parameters
    ----------
    galaxy : ``Galaxy class`` object
        The galaxy to project, centered and aligned (see
        ``galaxychop.preproc.center`` and ``galaxychop.preproc.star_align``).
    projection : str, default "edge-on"
        ``"edge-on"`` (x-z plane, line of sight ``vy``) or ``"face-on"``
        (x-y plane, line of sight ``vz``).
    ptypes : tuple of str, default ("stars",)
        Particle types to project, pooled together. ``None`` means all.
    bins : int, default 50
        Number of pixels per side.
    half_width : float, optional
        Half the side of the map, in kpc. By default the 99th percentile of
        the projected distance to the origin of the particles of ``ptypes``.
    min_particles : int, default 1
        Pixels with fewer particles get ``NaN`` velocities: with a handful
        of particles they are noise, much like a signal-to-noise cut.
    components : str or tuple of str, optional
        Only for decomposed galaxies: component labels (e.g. ``"Disk"``)
        whose particles are projected, pooled together. The default
        ``half_width`` still comes from all the particles of ``ptypes``, so
        every component shares the grid and their masses add up to the
        whole galaxy map.

    Returns
    -------
    pandas.DataFrame
        One row per pixel (horizontal axis running first), with the pixel
        center coordinates (e.g. ``x`` and ``z`` for ``"edge-on"``) and the
        quantities above. Its ``attrs`` hold the grid geometry and the
        parameters used.

    Examples
    --------
    >>> maps = gchop.core.maps.projected_maps(galaxy, projection="face-on")
    >>> maps.pivot(index="y", columns="x", values="v_los")

    """
    if projection not in PROJECTIONS:
        raise ValueError(
            f"'projection' must be one of {list(PROJECTIONS)}, "
            f"found {projection!r}"
        )
    if not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise ValueError(f"'bins' must be a positive integer, found {bins!r}")
    if half_width is not None and half_width <= 0:
        raise ValueError(f"'half_width' must be positive, found {half_width}")
    if min_particles < 1:
        raise ValueError(
            f"'min_particles' must be >= 1, found {min_particles}"
        )
    if isinstance(components, str):
        components = (components,)

    h_axis, v_axis, vlos_axis = PROJECTIONS[projection]
    attributes = ["m", h_axis, v_axis, vlos_axis]
    if components is not None:
        if not hasattr(galaxy, "unique_components_labels"):
            raise TypeError("'components' needs a decomposed galaxy")
        unknown = set(components) - galaxy.unique_components_labels
        if unknown:
            raise ValueError(f"Unknown components {sorted(unknown)}")
        attributes.append("labels")

    df = galaxy.to_dataframe(
        ptypes=ptypes, attributes=attributes, galaxy_circular_velocity=False
    )

    # the default grid comes from every particle of ptypes, before keeping
    # only some components, so all the component maps share it
    if half_width is None:
        half_width = np.percentile(
            np.hypot(df[h_axis], df[v_axis]), DEFAULT_HALF_WIDTH_PCT
        )
    half_width = float(half_width)

    if components is not None:
        df = df[df["labels"].isin(components)]
    if df.empty:
        raise ValueError("No particles to project")

    m, h, v, vlos = (df[c].to_numpy() for c in attributes[:4])

    # flat pixel index of every particle (horizontal axis running first);
    # the outer edge belongs to the last pixel, as in numpy.histogram
    pixel_size = 2 * half_width / bins
    ih = np.minimum((h + half_width) // pixel_size, bins - 1)
    iv = np.minimum((v + half_width) // pixel_size, bins - 1)
    inside = (np.abs(h) <= half_width) & (np.abs(v) <= half_width)
    pixel = (iv * bins + ih)[inside].astype(int)
    m, vlos = m[inside], vlos[inside]

    def per_pixel(weights=None):
        return np.bincount(pixel, weights=weights, minlength=bins * bins)

    n_particles = per_pixel()
    mass = per_pixel(m)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_los = per_pixel(m * vlos) / mass
        # around each pixel's mean, instead of <v^2> - <v>^2, which loses
        # precision when the dispersion is small next to v_los
        sigma_los = np.sqrt(per_pixel(m * (vlos - v_los[pixel]) ** 2) / mass)
    sparse = n_particles < min_particles
    v_los[sparse] = sigma_los[sparse] = np.nan

    centers = -half_width + (np.arange(bins) + 0.5) * pixel_size
    hh, vv = np.meshgrid(centers, centers)

    maps = pd.DataFrame(
        {
            h_axis: hh.ravel(),
            v_axis: vv.ravel(),
            "n_particles": n_particles,
            "mass": mass,
            "surface_density": mass / pixel_size**2,
            "v_los": v_los,
            "sigma_los": sigma_los,
        }
    )
    maps.attrs.update(
        projection=projection,
        bins=bins,
        half_width=half_width,
        pixel_size=pixel_size,
        min_particles=min_particles,
        components=components,
    )
    return maps
