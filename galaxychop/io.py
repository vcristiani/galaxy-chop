# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Module galaxy-chop."""

# =============================================================================
# IMPORTS
# =============================================================================

import platform
import sys
from datetime import datetime, timezone
import json

from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table

import attr

import h5py

import numpy as np

from . import core
from .constants import VERSION

# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_H5_METADATA = {
    "GalaxyChop": VERSION,
    "author_email": "valeria.cristiani@unc.edu.ar",
    "affiliation": "IATE-OAC-CONICET",
    "url": "https://github.com/vcristiani/galaxy-chop/",
    "platform": platform.platform(),
    "system_encoding": sys.getfilesystemencoding(),
    "Python": sys.version,
    "format_version": 2.0,
}

FALLBACK_VERSION = 1.0

# =============================================================================
# UTILS
# =============================================================================


def _table_to_dict(table, key_suffix):
    kws = {f"{k}_{key_suffix}": v for k, v in table.items() if k != "id"}
    kws[f"potential_{key_suffix}"] = kws.pop(f"potential_{key_suffix}", None)
    return kws


def _df_to_table(df, ptype):
    table_df = df[df.ptype == ptype.humanize()]
    del table_df["ptype"]
    return Table.from_pandas(table_df)


# =============================================================================
# HDF 5
# =============================================================================

_READ_HDF5_VERSIONS = {}


def _register_read_hdf5(version):
    def dec(func):
        _READ_HDF5_VERSIONS[version] = func

    return dec


@_register_read_hdf5(1.0)
def _read_hdf5(
    stream, *, softening_s: float, softening_dm: float, softening_g: float
):
    star_table = Table.read(stream["stars"])
    dark_table = Table.read(stream["dark_matter"])
    gas_table = Table.read(stream["gas"])

    galaxy_kws = {
        "softening_s": softening_s,
        "softening_dm": softening_dm,
        "softening_g": softening_g,
    }

    star_kws = _table_to_dict(star_table, "s")
    galaxy_kws.update(star_kws)

    dark_kws = _table_to_dict(dark_table, "dm")
    galaxy_kws.update(dark_kws)

    gas_kws = _table_to_dict(gas_table, "g")
    galaxy_kws.update(gas_kws)

    galaxy = core.mkgalaxy(**galaxy_kws)

    return galaxy


@_register_read_hdf5(2.0)
def _read_hdf5(
    stream, *, softening_s: float, softening_dm: float, softening_g: float
):
    star_table = Table.read(stream["stars"])
    dark_table = Table.read(stream["dark_matter"])
    gas_table = Table.read(stream["gas"])

    galaxy_kws = {
        "softening_s": softening_s,
        "softening_dm": softening_dm,
        "softening_g": softening_g,
    }

    star_kws = _table_to_dict(star_table, "s")
    galaxy_kws.update(star_kws)

    dark_kws = _table_to_dict(dark_table, "dm")
    galaxy_kws.update(dark_kws)

    gas_kws = _table_to_dict(gas_table, "g")
    galaxy_kws.update(gas_kws)

    galaxy = core.mkgalaxy(**galaxy_kws)

    return galaxy


def read_hdf5(
    path_or_stream,
    *,
    softening_s: float = 0.0,
    softening_dm: float = 0.0,
    softening_g: float = 0.0,
):
    """
    h5py file reader.

    Reads the file containing masses, positions, velocities of stellar, dark
    matter and gas particles, and constructs a galaxy object. The file may
    include particle potentials. The softening value can be included.

    Parameters
    ----------
    path_or_stream : str or file-like
        Path to the h5 file containing the properties of the galaxy particles.
    softening_s : float, default value = 0
        Softening radius of star particles.
    softening_dm : float, default value = 0
        Softening radius of dark matter particles.
    softening_g : float, default value = 0
        Softening radius of gas particles.

    Returns
    -------
    galaxy : ``Galaxy class`` object.

    """
    with h5py.File(path_or_stream, "r") as f:
        version = f.attrs.get("format_version", FALLBACK_VERSION)
        parser = _READ_HDF5_VERSIONS[version]
        return parser(
            f,
            softening_s=softening_s,
            softening_dm=softening_dm,
            softening_g=softening_g,
        )


# WRITE =======================================================================


def to_hdf5(
    path_or_stream,
    galaxy,
    *,
    metadata=None,
    group=None,
    force_group=False,
    **kwargs,
):
    """
    HDF5 file writer.

    It is responsible for storing a galaxy in HDF5 format. The procedure only
    stores the attributes ``m``, ``x``, ``y``, ``z``, ``vx``, ``vy`` and
    ``vz``,  since all the other attributes can be derived from these, and
    the ``softenings`` can be arbitrarily changed at the galaxy
    creation/reading process

    Parameters
    ----------
    path_or_stream : str or file-like
        Path or file like object to the h5 to store the galaxy.
    galaxy : galaxychop.core.Galaxy
        The galaxy to store.
    metadata : dict or None (default None)
        Extra metadata to store in the h5 file.
    group : str or None (default None)
        HDF5 group name to store the galaxy data. If None, defaults to "galaxy".
    force_group : bool (default False)
        If True, deletes the group if it already exists.
        If False, raises ValueError when group exists.
    kwargs :
        Extra arguments to the function
        ``astropy.io.misc.hdf5.write_table_hdf5()``

    """
    # Use the _gchop_h5_ method to get metadata and particle set data
    gal_meta, psets = galaxy._gchop_h5_()

    # Set default group name if not provided
    group = "galaxy" if group is None else group

    # prepare global metadata
    h5_metadata = _DEFAULT_H5_METADATA.copy()
    h5_metadata["utc_timestamp"] = datetime.now(timezone.utc).isoformat()

    # prepare galaxy metadata
    gal_meta["user_metadata"] = json.dumps(metadata or {})

    # prepare kwargs
    kwargs.setdefault("append", True)
    kwargs.setdefault("overwrite", True)
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 9)

    with h5py.File(path_or_stream, "a") as h5:
        # Check if group already exists
        if group in h5 and force_group:
            del h5[group]
        elif group in h5:
            raise ValueError(
                f"Group '{group}' already exists in the HDF5 file"
            )

        for pset_name, (pset_meta, pset_table) in psets.items():
            pset_path = "/".join([group, pset_name])

            # write the tables
            write_table_hdf5(pset_table, h5, path=pset_path, **kwargs)
            h5[pset_path].attrs.update(pset_meta)

        # Store galaxy-level metadata on the galaxy group
        h5[group].attrs.update(gal_meta)

        # Store global metadata at root level
        h5.attrs.update(h5_metadata)


# =============================================================================
# NUMPY
# =============================================================================


def read_npy(
    path_or_stream_star,
    path_or_stream_dark,
    path_or_stream_gas,
    columns,
    path_or_stream_pot_s=None,
    path_or_stream_pot_dm=None,
    path_or_stream_pot_g=None,
    softening_s: float = 0.0,
    softening_dm: float = 0.0,
    softening_g: float = 0.0,
):
    """
    Npy file reader.

    Reads npy files containing the masses, positions and velocities of stellar
    particles, dark matter and gas particles, and constructs a galaxy object.
    Files containing particle potentials can be included. The softening value
    can be included.

    Parameters
    ----------
    path_or_stream_star : str or file like
        Path to the npy file containing the properties of the star particles.
    path_or_stream_dark : str or file like
        Path to the npy file containing the properties of the dark matter
        particles.
    path_or_stream_gas : str or file like
        Path to the npy file containing the properties of the gas particles.
    columns: list
        Specify column names.
    path_or_stream_pot_s : str or file like
        Path to the npy file containing the potentials of the star particles.
    path_or_stream_pot_dm : str or file like
        Path to the npy file containing the potentials of the dark matter
        particles.
    path_or_stream_pot_g : str or file like
        Path to the npy file containing the potentials of the gas particles.
    softening_s : float, default value = 0
        Softening radius of star particles.
    softening_dm : float, default value = 0
        Softening radius of dark matter particles.
    softening_g : float, default value = 0
        Softening radius of gas particles.

    Returns
    -------
    galaxy : ``Galaxy class`` object.

    """
    particles_star = np.load(path_or_stream_star)
    particles_dark = np.load(path_or_stream_dark)
    particles_gas = np.load(path_or_stream_gas)

    star_table = Table(particles_star, names=columns)
    dark_table = Table(particles_dark, names=columns)
    gas_table = Table(particles_gas, names=columns)

    if path_or_stream_pot_s is not None:
        pot_s = np.load(path_or_stream_pot_s)
        star_table.add_column(pot_s, name="potential")

    if path_or_stream_pot_dm is not None:
        pot_dm = np.load(path_or_stream_pot_dm)
        dark_table.add_column(pot_dm, name="potential")

    if path_or_stream_pot_g is not None:
        pot_g = np.load(path_or_stream_pot_g)
        gas_table.add_column(pot_g, name="potential")

    galaxy_kws = {
        "softening_s": softening_s,
        "softening_dm": softening_dm,
        "softening_g": softening_g,
    }

    star_kws = _table_to_dict(star_table, "s")
    galaxy_kws.update(star_kws)

    dark_kws = _table_to_dict(dark_table, "dm")
    galaxy_kws.update(dark_kws)

    gas_kws = _table_to_dict(gas_table, "g")
    galaxy_kws.update(gas_kws)

    galaxy = core.mkgalaxy(**galaxy_kws)

    return galaxy
