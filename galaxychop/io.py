# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Input/Output module for GalaxyChop.

This module provides functionality to read and write galaxy data in different
formats. It supports:

- **HDF5 format**: The primary format for storing galaxy data with full metadata
  and versioning support (versions 1.0 and 2.0)
- **NumPy format**: Legacy support for reading old datasets from `.npy` files

The module handles both simple `Galaxy` objects and `DecomposedGalaxy` objects
with component decomposition data.

Main Functions
--------------
read_hdf5 : function
    Read galaxy data from HDF5 files.
to_hdf5 : function
    Write galaxy data to HDF5 files.
read_npy : function
    Read galaxy data from NumPy files (legacy format).

Notes
-----
The HDF5 format is recommended for all new data storage as it includes:

- Complete metadata about the galaxy and particles
- Version information for backwards compatibility
- Support for decomposed galaxies with probabilities
- Efficient compression and storage

Examples
--------
Reading a galaxy from HDF5:

>>> import galaxychop as gchop
>>> galaxy = gchop.io.read_hdf5("my_galaxy.h5")

Writing a galaxy to HDF5:

>>> gchop.io.to_hdf5("output.h5", galaxy, metadata={"simulation": "EAGLE"})

Reading from NumPy files (legacy):

>>> galaxy = gchop.io.read_npy(
...     "stars.npy", "dm.npy", "gas.npy",
...     columns=["m", "x", "y", "z", "vx", "vy", "vz"]
... )
"""

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
from .models import DecomposedGalaxy, ComponentParticleSet

# =============================================================================
# CONSTANTS
# =============================================================================

#: Default metadata written to root level of HDF5 files, including version, author, and platform info.
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

#: Fallback format version (1.0) assumed when reading HDF5 files without explicit version metadata.
FALLBACK_VERSION = 1.0

# =============================================================================
# UTILS
# =============================================================================


def _table_to_dict(table, key_suffix):
    """Convert Astropy Table to dictionary with suffixed column keys for Galaxy construction."""
    kws = {f"{k}_{key_suffix}": v for k, v in table.items() if k != "id"}
    kws[f"potential_{key_suffix}"] = kws.pop(f"potential_{key_suffix}", None)
    return kws


def _df_to_table(df, ptype):
    """Filter DataFrame by particle type and convert to Astropy Table (for legacy data conversion)."""
    table_df = df[df.ptype == ptype.humanize()]
    del table_df["ptype"]
    return Table.from_pandas(table_df)


# =============================================================================
# HDF 5
# =============================================================================

#: Registry mapping format versions to reader classes, populated by @_register_read_hdf5 decorator.
_READ_HDF5_VERSIONS = {}


def _register_read_hdf5(version):
    """Decorator that registers an HDF5 reader class for a specific format version."""
    def dec(cls):
        _READ_HDF5_VERSIONS[version] = cls
        return cls

    return dec


class GalaxyHDF5ReaderABC:
    """Abstract base class for HDF5 galaxy file readers.

    This class defines the interface that all HDF5 readers must implement.
    Different reader versions handle different HDF5 format versions to ensure
    backwards compatibility.

    Notes
    -----
    Concrete reader classes should:

    1. Inherit from this class
    2. Be decorated with ``@_register_read_hdf5(version)``
    3. Implement the ``read`` method

    See Also
    --------
    HDF5ReaderV1 : Reader for format version 1.0
    HDF5ReaderV2 : Reader for format version 2.0
    """

    def read(self, stream, **kwargs):  # type: ignore
        """Parse the HDF5 stream and return a galaxy object.

        Parameters
        ----------
        stream : h5py.File
            Opened HDF5 file object.
        **kwargs
            Additional keyword arguments specific to each reader version.

        Returns
        -------
        core.Galaxy or models.DecomposedGalaxy
            Reconstructed galaxy object from the HDF5 data.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement read")


@_register_read_hdf5(1.0)
class HDF5ReaderV1(GalaxyHDF5ReaderABC):
    """HDF5 reader for format version 1.0 (legacy format).

    This reader handles HDF5 files created with the original GalaxyChop format,
    where particle data was stored in three separate datasets at the root level:
    ``stars``, ``dark_matter``, and ``gas``.

    Format version 1.0 files contain only basic ``Galaxy`` objects without
    decomposition information.

    Parameters
    ----------
    None

    Notes
    -----
    Version 1.0 format structure:

    - ``/stars`` : HDF5 dataset with stellar particle data
    - ``/dark_matter`` : HDF5 dataset with dark matter particle data
    - ``/gas`` : HDF5 dataset with gas particle data

    This format is maintained for backwards compatibility with older files.
    New files should use format version 2.0.

    See Also
    --------
    HDF5ReaderV2 : Reader for the current format version 2.0
    """

    def read(
        self,
        stream,
        *,
        softening_s: float = 0,
        softening_dm: float = 0,
        softening_g: float = 0,
    ):
        """Read a Galaxy from an HDF5 file in version 1.0 format.

        Parameters
        ----------
        stream : h5py.File
            Opened HDF5 file object.
        softening_s : float, default=0
            Softening length for star particles.
        softening_dm : float, default=0
            Softening length for dark matter particles.
        softening_g : float, default=0
            Softening length for gas particles.

        Returns
        -------
        core.Galaxy
            Galaxy object reconstructed from the HDF5 data.
        """
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
class HDF5ReaderV2(GalaxyHDF5ReaderABC):
    """HDF5 reader for format version 2.0 (current format).

    This reader handles HDF5 files in the current GalaxyChop format, which uses
    a hierarchical structure with groups. It supports both simple ``Galaxy``
    objects and ``DecomposedGalaxy`` objects with component probabilities.

    Format version 2.0 features:

    - Hierarchical group structure (e.g., ``/galaxy/stars``)
    - Support for multiple galaxies in a single file
    - Particle-level and galaxy-level metadata
    - Component decomposition with probabilities

    Notes
    -----
    Version 2.0 format structure:

    - ``/group_name/stars`` : HDF5 dataset with stellar particle data
    - ``/group_name/dark_matter`` : HDF5 dataset with dark matter data
    - ``/group_name/gas`` : HDF5 dataset with gas particle data
    - ``/group_name`` (attrs) : Galaxy-level metadata
    - ``/`` (attrs) : File-level metadata

    See Also
    --------
    HDF5ReaderV1 : Reader for legacy format version 1.0
    """

    def _galaxy_builder(
        self,
        *,
        stars_dataset,
        dark_matter_dataset,
        gas_dataset,
        gal_meta,
        softening_s,
        softening_dm,
        softening_g,
    ):
        """Construct a basic Galaxy from HDF5 particle datasets without decomposition information."""

        ds_and_soft = zip(
            [softening_s, softening_dm, softening_g],
            [stars_dataset, dark_matter_dataset, gas_dataset],
        )

        psets = {}
        for softening, dataset in ds_and_soft:

            table = Table.read(dataset)
            meta = dict(dataset.attrs)

            ptype = core.ParticleSetType.mktype(meta["ptype"])

            kws = {
                f"{k}": v
                for k, v in table.items()
                if k != "id"
                and not (k.endswith(".mask") or k.startswith("probabilities"))
            }

            del table, meta

            pset = core.ParticleSet(
                ptype=ptype,
                softening=softening,
                **kws,
            )

            psets[ptype.name.lower()] = pset

        gal = core.Galaxy(**psets)

        return gal

    def _decomposed_galaxy_builder(
        self,
        *,
        stars_dataset,
        dark_matter_dataset,
        gas_dataset,
        gal_meta,
        softening_s,
        softening_dm,
        softening_g,
    ):
        """Construct a DecomposedGalaxy from HDF5 datasets including component probabilities."""

        method = gal_meta["method"]
        component_name_mapping = json.loads(gal_meta["component_name_mapping"])

        ds_and_soft = zip(
            [softening_s, softening_dm, softening_g],
            [stars_dataset, dark_matter_dataset, gas_dataset],
        )

        psets = {}
        for softening, dataset in ds_and_soft:

            table = Table.read(dataset)
            meta = dict(dataset.attrs)

            ptype = core.ParticleSetType.mktype(meta["ptype"])

            kws = {
                f"{k}": v
                for k, v in table.items()
                if k != "id"
                and not (k.endswith(".mask") or k.startswith("probabilities"))
            }

            has_probabilities = meta["has_probabilities"]
            probabilities_n = (
                meta["probabilities_n"] if has_probabilities else 1
            )

            probabilities_columns = [
                f"probabilities_{n}" for n in range(probabilities_n)
            ]
            probabilities = table[probabilities_columns].to_pandas().values

            del table, meta

            pset = ComponentParticleSet(
                ptype=ptype,
                softening=softening,
                probabilities=probabilities,
                **kws,
            )

            psets[ptype.name.lower()] = pset

        gal = DecomposedGalaxy(
            method=method,
            component_name_mapping=component_name_mapping,
            **psets,
        )

        return gal

    def read(
        self,
        stream,
        *,
        group=None,
        softening_s: float = 0,
        softening_dm: float = 0,
        softening_g: float = 0,
    ):
        """Read a Galaxy or DecomposedGalaxy from an HDF5 file in version 2.0 format.

        This method automatically detects the galaxy type from metadata and uses
        the appropriate builder method to reconstruct the object.

        Parameters
        ----------
        stream : h5py.File
            Opened HDF5 file object.
        group : str, optional
            HDF5 group name where the galaxy data is stored. If None, defaults
            to "galaxy". This allows multiple galaxies to be stored in different
            groups within the same file.
        softening_s : float, default=0
            Softening length for star particles.
        softening_dm : float, default=0
            Softening length for dark matter particles.
        softening_g : float, default=0
            Softening length for gas particles.

        Returns
        -------
        core.Galaxy or models.DecomposedGalaxy
            Galaxy object reconstructed from the HDF5 data. The specific type
            depends on the galaxy_type metadata stored in the file.

        Raises
        ------
        ValueError
            If the galaxy_type in the file metadata is not recognized.
        """
        # Registry mapping galaxy types to their builders
        galaxy_builders = {
            DecomposedGalaxy.__name__: self._decomposed_galaxy_builder,
            core.Galaxy.__name__: self._galaxy_builder,
        }

        # Set default group name if not provided
        group = "galaxy" if group is None else group

        gal_meta = dict(stream[group].attrs)
        gal_type = gal_meta["galaxy_type"]

        try:
            builder = galaxy_builders[gal_type]
        except KeyError:
            raise ValueError(f"Unknown galaxy type {gal_type}")

        stars_dataset = stream[f"{group}/stars"]
        dark_matter_dataset = stream[f"{group}/dark_matter"]
        gas_dataset = stream[f"{group}/gas"]

        galaxy = builder(
            stars_dataset=stars_dataset,
            dark_matter_dataset=dark_matter_dataset,
            gas_dataset=gas_dataset,
            gal_meta=gal_meta,
            softening_s=softening_s,
            softening_dm=softening_dm,
            softening_g=softening_g,
        )

        return galaxy


def read_hdf5(
    path_or_stream,
    softening_s: float = 0,
    softening_dm: float = 0,
    softening_g: float = 0,
    **kwargs,
):
    """Read galaxy data from an HDF5 file.

    This is the main function for loading galaxy data stored in HDF5 format.
    It automatically detects the file format version and uses the appropriate
    reader to reconstruct the galaxy object.

    The function supports both legacy format (version 1.0) and current format
    (version 2.0), and can read both simple ``Galaxy`` objects and
    ``DecomposedGalaxy`` objects with component probabilities.

    Parameters
    ----------
    path_or_stream : str or file-like
        Path to the HDF5 file, or an open file-like object. If a string path
        is provided, the file will be opened and closed automatically.
    softening_s : float, default=0
        Softening length for star particles. This value is used in force
        calculations to prevent numerical singularities at small distances.
    softening_dm : float, default=0
        Softening length for dark matter particles.
    softening_g : float, default=0
        Softening length for gas particles.
    **kwargs
        Additional keyword arguments passed to the format-specific reader.
        For version 2.0 files, you can specify:

        - ``group`` (str): HDF5 group name to read from (default: "galaxy")

    Returns
    -------
    core.Galaxy or models.DecomposedGalaxy
        Reconstructed galaxy object. The specific type depends on what was
        stored in the file.

    Raises
    ------
    KeyError
        If the file format version is not supported.
    ValueError
        If the file contains an unrecognized galaxy type (version 2.0 only).

    Notes
    -----
    The function automatically detects the format version from the file's
    ``format_version`` attribute. If this attribute is missing, version 1.0
    is assumed for backwards compatibility.

    Softening lengths are not stored in the HDF5 files and must be provided
    at read time. They can be set to different values than when the galaxy
    was originally saved.

    Examples
    --------
    Read a galaxy from an HDF5 file:

    >>> import galaxychop as gchop
    >>> galaxy = gchop.io.read_hdf5("my_galaxy.h5")

    Read with specific softening lengths:

    >>> galaxy = gchop.io.read_hdf5(
    ...     "my_galaxy.h5",
    ...     softening_s=0.1,
    ...     softening_dm=0.2,
    ...     softening_g=0.1
    ... )

    Read from a specific group in a version 2.0 file:

    >>> galaxy = gchop.io.read_hdf5("multi_galaxy.h5", group="galaxy_1")

    See Also
    --------
    to_hdf5 : Write galaxy data to HDF5 format
    read_npy : Read galaxy data from NumPy files (legacy)
    """
    with h5py.File(path_or_stream, "r") as f:
        version = f.attrs.get("format_version", FALLBACK_VERSION)
        parser_class = _READ_HDF5_VERSIONS[version]
        parser = parser_class()
        return parser.read(
            f,
            softening_s=softening_s,
            softening_dm=softening_dm,
            softening_g=softening_g,
            **kwargs,
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
    """Write galaxy data to an HDF5 file in format version 2.0.

    This function saves galaxy data to HDF5 format with a hierarchical structure.
    It stores particle data (mass, positions, velocities, and optionally potentials)
    along with comprehensive metadata about the galaxy, decomposition method (if
    applicable), and the GalaxyChop environment.

    The function supports both ``Galaxy`` and ``DecomposedGalaxy`` objects. For
    decomposed galaxies, component probabilities are stored as well.

    Parameters
    ----------
    path_or_stream : str or file-like
        Path to the HDF5 file to create/modify, or an open file-like object.
        If the file doesn't exist, it will be created. If it exists, data will
        be added to it (see ``force_group`` parameter).
    galaxy : core.Galaxy or models.DecomposedGalaxy
        The galaxy object to save. Can be either a basic Galaxy or a
        DecomposedGalaxy with component information.
    metadata : dict, optional
        Additional user-defined metadata to store in the file. This will be
        serialized as JSON and stored in the root-level ``user_metadata``
        attribute. Useful for storing information about simulations, parameters,
        or analysis details.
    group : str, optional
        HDF5 group name where galaxy data will be stored. If None, defaults to
        "galaxy". Using different group names allows multiple galaxies to be
        stored in the same file.
    force_group : bool, default=False
        Controls behavior when the specified group already exists:

        - If True: Deletes existing group and overwrites with new data
        - If False: Raises ValueError to prevent accidental data loss

    **kwargs
        Additional keyword arguments passed to ``astropy.io.misc.hdf5.write_table_hdf5()``.
        Common options include:

        - ``compression`` (str): Compression algorithm, default "gzip"
        - ``compression_opts`` (int): Compression level (0-9), default 9
        - ``overwrite`` (bool): Allow overwriting datasets, default True
        - ``append`` (bool): Allow appending to existing file, default True

    Raises
    ------
    ValueError
        If ``force_group=False`` and the specified group already exists in the file.

    Notes
    -----
    **Data Storage:**

    The function stores only fundamental particle properties (mass, position,
    velocity, and potential). Derived properties are not saved, as they can be
    recomputed from these fundamentals.

    Softening lengths are NOT stored in the file. They must be provided when
    reading the file via the ``softening_s``, ``softening_dm``, and ``softening_g``
    parameters of :func:`read_hdf5`.

    **File Structure (Format Version 2.0):**

    - ``/`` (root attributes): Global metadata, version info, timestamp
    - ``/group_name/`` (group attributes): Galaxy-level metadata
    - ``/group_name/stars``: Stellar particle dataset
    - ``/group_name/dark_matter``: Dark matter particle dataset
    - ``/group_name/gas``: Gas particle dataset

    **Compression:**

    By default, data is compressed using gzip with maximum compression level (9).
    This significantly reduces file size with minimal performance impact for
    typical use cases.

    Examples
    --------
    Save a basic galaxy:

    >>> import galaxychop as gchop
    >>> gchop.io.to_hdf5("my_galaxy.h5", galaxy)

    Save with custom metadata:

    >>> metadata = {
    ...     "simulation": "EAGLE",
    ...     "halo_id": 12345,
    ...     "redshift": 0.0
    ... }
    >>> gchop.io.to_hdf5("galaxy.h5", galaxy, metadata=metadata)

    Save multiple galaxies to the same file:

    >>> gchop.io.to_hdf5("galaxies.h5", galaxy1, group="galaxy_1")
    >>> gchop.io.to_hdf5("galaxies.h5", galaxy2, group="galaxy_2")

    Overwrite existing data:

    >>> gchop.io.to_hdf5("galaxy.h5", new_galaxy, force_group=True)

    Use minimal compression for faster I/O:

    >>> gchop.io.to_hdf5("galaxy.h5", galaxy, compression_opts=1)

    See Also
    --------
    read_hdf5 : Read galaxy data from HDF5 format
    core.Galaxy._gchop_h5_ : Internal method that prepares galaxy data for storage
    """
    # Use the _gchop_h5_ method to get metadata and particle set data
    gal_meta, psets = galaxy._gchop_h5_()

    # Set default group name if not provided
    group = "galaxy" if group is None else group

    # prepare global metadata
    h5_metadata = _DEFAULT_H5_METADATA.copy()
    h5_metadata["utc_timestamp"] = datetime.now(timezone.utc).isoformat()
    h5_metadata["user_metadata"] = json.dumps(metadata or {})

    # prepare galaxy metadata
    gal_meta["component_name_mapping"] = json.dumps(
        gal_meta["component_name_mapping"]
    )

    # prepare kwargs
    kwargs.setdefault("append", True)
    kwargs.setdefault("overwrite", True)
    kwargs.setdefault("compression", "gzip")
    kwargs.setdefault("serialize_meta", True)
    kwargs.setdefault("compression_opts", 9)

    with h5py.File(
        path_or_stream,
        "a",
    ) as h5:
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
    """Read galaxy data from NumPy .npy files (legacy format).

    This function provides backwards compatibility with older datasets stored
    as separate NumPy arrays. It reads particle data from multiple .npy files
    (one per particle type) and constructs a basic ``Galaxy`` object.

    This format is deprecated. For new projects, use :func:`to_hdf5` and
    :func:`read_hdf5` instead, which provide better metadata support, compression,
    and organization.

    Parameters
    ----------
    path_or_stream_star : str or file-like
        Path to the .npy file containing stellar particle data. The file should
        contain a 2D array where each row is a particle and columns correspond
        to the ``columns`` parameter.
    path_or_stream_dark : str or file-like
        Path to the .npy file containing dark matter particle data.
    path_or_stream_gas : str or file-like
        Path to the .npy file containing gas particle data.
    columns : list of str
        Column names for the particle properties in the arrays. Typically
        ``["m", "x", "y", "z", "vx", "vy", "vz"]`` for mass, 3D position,
        and 3D velocity. The order must match the column order in the arrays.
    path_or_stream_pot_s : str or file-like, optional
        Path to a separate .npy file containing stellar particle potentials.
        If provided, potentials will be added to the Galaxy.
    path_or_stream_pot_dm : str or file-like, optional
        Path to a separate .npy file containing dark matter particle potentials.
    path_or_stream_pot_g : str or file-like, optional
        Path to a separate .npy file containing gas particle potentials.
    softening_s : float, default=0.0
        Softening length for star particles.
    softening_dm : float, default=0.0
        Softening length for dark matter particles.
    softening_g : float, default=0.0
        Softening length for gas particles.

    Returns
    -------
    core.Galaxy
        Basic Galaxy object constructed from the NumPy arrays.

    Notes
    -----
    **Legacy Format:**

    This is a low-level utility function primarily intended for consuming old
    datasets. It constructs a simple ``Galaxy`` object from multiple NumPy
    arrays without any metadata or versioning support.

    **Limitations:**

    - No metadata storage
    - Requires separate files for each particle type and property
    - No support for decomposed galaxies
    - No compression
    - No versioning

    **Migration:**

    To convert old .npy files to the modern HDF5 format:

    >>> import galaxychop as gchop
    >>> # Read from old format
    >>> galaxy = gchop.io.read_npy(
    ...     "stars.npy", "dm.npy", "gas.npy",
    ...     columns=["m", "x", "y", "z", "vx", "vy", "vz"]
    ... )
    >>> # Save to new format
    >>> gchop.io.to_hdf5("galaxy.h5", galaxy)

    Examples
    --------
    Read particle data from separate NumPy files:

    >>> import galaxychop as gchop
    >>> galaxy = gchop.io.read_npy(
    ...     "stars.npy",
    ...     "dark_matter.npy",
    ...     "gas.npy",
    ...     columns=["m", "x", "y", "z", "vx", "vy", "vz"]
    ... )

    Include potentials from additional files:

    >>> galaxy = gchop.io.read_npy(
    ...     "stars.npy", "dm.npy", "gas.npy",
    ...     columns=["m", "x", "y", "z", "vx", "vy", "vz"],
    ...     path_or_stream_pot_s="star_potentials.npy",
    ...     path_or_stream_pot_dm="dm_potentials.npy",
    ...     path_or_stream_pot_g="gas_potentials.npy"
    ... )

    Specify softening lengths:

    >>> galaxy = gchop.io.read_npy(
    ...     "stars.npy", "dm.npy", "gas.npy",
    ...     columns=["m", "x", "y", "z", "vx", "vy", "vz"],
    ...     softening_s=0.1,
    ...     softening_dm=0.2,
    ...     softening_g=0.1
    ... )

    See Also
    --------
    read_hdf5 : Read galaxy data from HDF5 format (recommended)
    to_hdf5 : Write galaxy data to HDF5 format (recommended)
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
