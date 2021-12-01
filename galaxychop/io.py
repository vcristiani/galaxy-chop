# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Module galaxy-chop."""

# =============================================================================
# IMPORTS
# =============================================================================

import datetime as dt
import platform
import sys

from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.table import Table


import h5py


import numpy as np

from . import __version__ as VERSION
from . import data


# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_METADATA = {
    "GalaxyChop": VERSION,
    "author_email": "valeria.cristiani@unc.edu.ar ",
    "affiliation": "IATE-OAC-CONICET",
    "url": "https://github.com/vcristiani/galaxy-chop/",
    "platform": platform.platform(),
    "system_encoding": sys.getfilesystemencoding(),
    "Python": sys.version,
}


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


def read_hdf5(
    path_or_stream,
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
    path_or_stream : str
        Path to the h5 file containing the properties of the galaxy particles.
    softening_s : float, default value = 0
        Softening radius of star particles.
    softening_dm : float, default value = 0
        Softening radius of dark matter particles.
    softening_g : float, default value = 0
        Softening radius of gas particles.

    Returns
    -------
    galaxy : object of Galaxy class.
    """
    with h5py.File(path_or_stream, "r") as f:
        star_table = Table.read(f["stars"])
        dark_table = Table.read(f["dark_matter"])
        gas_table = Table.read(f["gas"])

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

    galaxy = data.mkgalaxy(**galaxy_kws)

    return galaxy


def to_hdf5(path_or_stream, galaxy, metadata=None, **kwargs):
    """HDF5 file writer.

    It is responsible for storing a galaxy in HDF5 format. The procedure only
    only stores the attributes ``m``, ``x``, ``y``, ``z``, ``vx``, ``vy`` and
    ``vz``,  since all the other attributes can be derived from these, and
    the ``softenings`` can be arbitrarily changed at the galaxy
    creation/reading process

    Parameters
    ----------
    path_or_stream : str
        Path to the h5 to store the galaxy.
    metadata : dict or None (default None)
        Extra metadata to store in the h5 file.
    kwargs :
        Extra arguments to the function
        ``astropy.io.misc.hdf5.write_table_hdf5()``


    """

    attributes = ["ptype", "m", "x", "y", "z", "vx", "vy", "vz"]
    if galaxy.has_potential_:
        attributes.append("potential")

    df = galaxy.to_dataframe(attributes=attributes)

    # create the id column for all the
    df.insert(0, "id", df.index.to_numpy())

    stars_table = _df_to_table(df, data.ParticleSetType.STARS)
    dm_table = _df_to_table(df, data.ParticleSetType.DARK_MATTER)
    gas_table = _df_to_table(df, data.ParticleSetType.GAS)

    # prepare metadata
    metadata = _DEFAULT_METADATA.copy()
    metadata["utc_timestamp"] = dt.datetime.utcnow().isoformat()
    metadata.update(metadata or {})

    # prepare kwargs
    kwargs.setdefault("append", True)
    kwargs.setdefault("overwrite", True)
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("compression_opts", 9)

    with h5py.File(path_or_stream, "a") as h5:
        write_table_hdf5(stars_table, h5, path="stars", **kwargs)
        write_table_hdf5(dm_table, h5, path="dark_matter", **kwargs)
        write_table_hdf5(gas_table, h5, path="gas", **kwargs)

        h5.attrs.update(metadata)


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
    path_or_stream_star : str
        Path to the npy file containing the properties of the star particles.
    path_or_stream_dark : str
        Path to the npy file containing the properties of the dark matter
        particles.
    path_or_stream_gas : str
        Path to the npy file containing the properties of the gas particles.
    columns: list
        Specify column names.
    path_or_stream_pot_s : str
        Path to the npy file containing the potentials of the star particles.
    path_or_stream_pot_dm : str
        Path to the npy file containing the potentials of the dark matter
        particles.
    path_or_stream_pot_g : str
        Path to the npy file containing the potentials of the gas particles.
    softening_s : float, default value = 0
        Softening radius of star particles.
    softening_dm : float, default value = 0
        Softening radius of dark matter particles.
    softening_g : float, default value = 0
        Softening radius of gas particles.

    Returns
    -------
    galaxy : object of Galaxy class.
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

    galaxy = data.mkgalaxy(**galaxy_kws)

    return galaxy
