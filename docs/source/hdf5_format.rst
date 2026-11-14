.. _hdf5-format:

HDF5 file format (version 2.0)
==============================

This page specifies the internal layout of the `HDF5
<https://www.hdfgroup.org/solutions/hdf5/>`_ files written by
:func:`galaxychop.io.to_hdf5` (and :meth:`Galaxy.to_hdf5()
<galaxychop.core.galaxy.Galaxy.to_hdf5>`), and read back by
:func:`galaxychop.io.read_hdf5`. Both plain galaxies
(:class:`~galaxychop.core.galaxy.Galaxy`) and decomposed galaxies
(:class:`~galaxychop.decomposers.core.decomposed_galaxy.DecomposedGalaxy`) share the same layout;
the latter only add some columns and attributes.

The files can be read with any HDF5 tool (``h5py``, ``h5dump``, HDFView, ...),
but :func:`~galaxychop.io.read_hdf5` is the supported way to rebuild a galaxy.


Layout
------

.. code-block:: text

    /                                    file attributes (format version, ...)
    └── galaxy/                          group: one galaxy, with its attributes
        ├── stars                        dataset: one row per star particle
        ├── dark_matter                  dataset: one row per dark matter particle
        ├── gas                          dataset: one row per gas particle

The group is named ``galaxy`` by default. Passing ``group="<name>"`` to
:func:`~galaxychop.io.to_hdf5` and :func:`~galaxychop.io.read_hdf5` stores and
reads several galaxies in the same file, one per group. Writing to an existing
group raises an error, unless ``force_group=True``, which replaces it.

The three particle datasets are always present, even if a particle type has no
particles (the dataset is then empty).

All the attributes are scalars: strings, booleans or numbers. Dictionaries
are stored as JSON strings. At the HDF5 level, strings are variable-length
UTF-8 strings, and booleans are 8-bit integer enums (``FALSE = 0``,
``TRUE = 1``), the way h5py writes them.


File attributes (``/``)
-----------------------

Written on every call to :func:`~galaxychop.io.to_hdf5`, so with several
groups they describe the last write.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Attribute
     - Type
     - Meaning
   * - ``format_version``
     - float
     - Version of this layout: ``2.0``. Files without it are read as the
       legacy version 1.0.
   * - ``GalaxyChop``
     - str
     - Version of GalaxyChop that wrote the file.
   * - ``utc_timestamp``
     - str
     - Date and time of the write, ISO 8601, UTC.
   * - ``user_metadata``
     - str (JSON)
     - The ``metadata`` dict passed to :func:`~galaxychop.io.to_hdf5`
       (``{}`` if none). Not read back by :func:`~galaxychop.io.read_hdf5`.
   * - ``Python``, ``platform``, ``system_encoding``
     - str
     - Environment that wrote the file.
   * - ``url``, ``author_email``, ``affiliation``
     - str
     - Information about the GalaxyChop project.


Galaxy attributes (``/galaxy``)
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Meaning
   * - ``galaxy_type``
     - str
     - ``"Galaxy"`` or ``"DecomposedGalaxy"``. Decides which object
       :func:`~galaxychop.io.read_hdf5` rebuilds.
   * - ``has_potential``
     - bool
     - Whether the particles have their specific potential energy.

Decomposed galaxies only:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Meaning
   * - ``method``
     - str
     - Name of the decomposer that made the decomposition (e.g.
       ``"JHistogram"``).
   * - ``component_name_mapping``
     - str (JSON)
     - Physical name of each component number, e.g.
       ``{"0": "Spheroid", "1": "Disk"}`` (JSON keys are strings; they are
       turned back into integers when read). ``{}`` when the components have
       no physical name (e.g. ``KMeans``).
   * - ``has_probabilities``
     - bool
     - Whether the decomposition is probabilistic (it gives every star a
       membership probability for every component).


Particle datasets (``/galaxy/stars``, ``/galaxy/dark_matter``, ``/galaxy/gas``)
-------------------------------------------------------------------------------

Each dataset is a table (an HDF5 compound dataset) written with
:func:`astropy.io.misc.hdf5.write_table_hdf5`, gzip compressed (level 9 by
default), with one row per particle.

Only the fundamental properties are stored. Everything else (energies,
angular momentum, stellar dynamics, ...) is recomputed from them. The
**softening is not stored**: it is given when reading, with the
``softening_s``, ``softening_dm`` and ``softening_g`` parameters of
:func:`~galaxychop.io.read_hdf5` (``0`` by default).

The columns carry no units: they are always in the units below, the ones
GalaxyChop uses internally.

.. list-table::
   :header-rows: 1
   :widths: 20 12 23 45

   * - Column
     - Type
     - Unit
     - Meaning
   * - ``id``
     - int64
     -
     - Index of the particle within its type (``0`` to ``n - 1``). Not read
       back.
   * - ``m``
     - float64
     - M\ :sub:`☉`
     - Mass.
   * - ``x``, ``y``, ``z``
     - float64
     - kpc
     - Position.
   * - ``vx``, ``vy``, ``vz``
     - float64
     - km s\ :sup:`-1`
     - Velocity.
   * - ``potential``
     - float64
     - (km s\ :sup:`-1`)\ :sup:`2`
     - Specific potential energy. **Only present if** ``has_potential`` is
       true.

Decomposed galaxies add:

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Column
     - Type
     - Meaning
   * - ``components``
     - float64
     - Component number assigned to the particle (``0``, ``1``, ...).
       Masked (see ``components.mask``) for the particles without a
       component: every dark matter and gas particle, and the stars the
       decomposer could not classify.
   * - ``components.mask``
     - bool
     - ``True`` where ``components`` has no value. Written by astropy to
       store the masked column.
   * - ``labels``
     - fixed-length string
     - Readable name of the component: the name from
       ``component_name_mapping`` (e.g. ``Disk``), else the component number
       as text (e.g. ``0``); for the particles without a component, their
       particle type (``stars``, ``dark_matter`` or ``gas``). The length
       of the string is that of the longest label, so it changes from file
       to file; shorter labels are padded with null bytes.
   * - ``probabilities_0`` … ``probabilities_{k-1}``
     - float64
     - Probability that the particle belongs to component ``0`` … ``k-1``.
       ``NaN`` for the particles without a component. Deterministic
       decompositions (``has_probabilities`` false) still store one
       ``probabilities_0`` column, full of ``NaN``.


Dataset attributes
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Meaning
   * - ``ptype``
     - str
     - Particle type: ``"STARS"``, ``"DARK_MATTER"`` or ``"GAS"``.
   * - ``pset_type``
     - str
     - ``"ParticleSet"``, or ``"DecomposedParticleSet"`` for decomposed
       galaxies.
   * - ``has_potential``
     - bool
     - Whether the ``potential`` column is present.
   * - ``has_probabilities``
     - bool
     - Decomposed galaxies only: same as the galaxy attribute.
   * - ``probabilities_n``
     - int
     - Decomposed galaxies only: number ``k`` of ``probabilities_*`` columns
       to read (``0`` for deterministic decompositions).


Astropy metadata (``<ptype>.__table_column_meta__``)
----------------------------------------------------

Next to every particle dataset, astropy stores a dataset with the YAML
description of the table (the type of each column and how the masked
``components`` column is serialized). It lets
:obj:`astropy.table.Table.read` rebuild the table exactly; other tools can
ignore it.


Examples
--------

These examples read a decomposed galaxy without GalaxyChop: the format
version, the decomposition method, and the ``x`` coordinate and label of
every star. HDF5 converts a compound dataset by field name, so a program
can read only the fields it needs.

Python
~~~~~~

With `h5py <https://www.h5py.org/>`_:

.. code-block:: python

    import json

    import h5py

    with h5py.File("decomposed.h5", "r") as h5:
        print(h5.attrs["format_version"])  # 2.0

        galaxy = h5["galaxy"]
        print(galaxy.attrs["method"])
        names = json.loads(galaxy.attrs["component_name_mapping"])

        stars = galaxy["stars"][()]  # numpy structured array
        x, labels = stars["x"], stars["labels"].astype(str)


R
~

With `hdf5r <https://cran.r-project.org/package=hdf5r>`_ and
`jsonlite <https://cran.r-project.org/package=jsonlite>`_:

.. code-block:: r

    library(hdf5r)

    h5 <- H5File$new("decomposed.h5", mode = "r")
    print(h5attr(h5, "format_version"))  # 2

    galaxy <- h5[["galaxy"]]
    print(h5attr(galaxy, "method"))
    names <- jsonlite::fromJSON(h5attr(galaxy, "component_name_mapping"))

    stars <- galaxy[["stars"]]$read()  # data.frame, one column per field
    x <- stars$x
    labels <- stars$labels

    h5$close_all()


Julia
~~~~~

With `HDF5.jl <https://juliaio.github.io/HDF5.jl/stable/>`_ and
`JSON.jl <https://github.com/JuliaIO/JSON.jl>`_:

.. code-block:: julia

    using HDF5, JSON

    h5open("decomposed.h5", "r") do h5
        println(read_attribute(h5, "format_version"))  # 2.0

        galaxy = h5["galaxy"]
        println(read_attribute(galaxy, "method"))
        names = JSON.parse(read_attribute(galaxy, "component_name_mapping"))

        stars = read(galaxy["stars"])  # Vector of NamedTuples, one per star
        x = [star.x for star in stars]
        labels = [star.labels for star in stars]
    end


C
~

With the HDF5 C library (compile with ``h5cc read_galaxy.c``):

.. code-block:: c

    #include <stdio.h>
    #include <stdlib.h>
    #include <hdf5.h>

    int main(void) {
        hid_t file = H5Fopen("decomposed.h5", H5F_ACC_RDONLY, H5P_DEFAULT);

        /* format version: a double attribute of the root */
        double version;
        hid_t attr = H5Aopen(file, "format_version", H5P_DEFAULT);
        H5Aread(attr, H5T_NATIVE_DOUBLE, &version);
        H5Aclose(attr);
        printf("format version: %.1f\n", version);

        /* decomposition method: a variable-length string attribute, read
           with its own type into a char* that HDF5 allocates */
        hid_t galaxy = H5Gopen(file, "galaxy", H5P_DEFAULT);
        attr = H5Aopen(galaxy, "method", H5P_DEFAULT);
        hid_t attr_type = H5Aget_type(attr);
        char *method;
        H5Aread(attr, attr_type, &method);
        printf("method: %s\n", method);
        H5free_memory(method);
        H5Tclose(attr_type);
        H5Aclose(attr);

        hid_t stars = H5Dopen(galaxy, "stars", H5P_DEFAULT);
        hid_t space = H5Dget_space(stars);
        hssize_t n = H5Sget_simple_extent_npoints(space);

        /* the "x" field alone: a compound type with only that member */
        hid_t x_type = H5Tcreate(H5T_COMPOUND, sizeof(double));
        H5Tinsert(x_type, "x", 0, H5T_NATIVE_DOUBLE);
        double *x = malloc(n * sizeof(double));
        H5Dread(stars, x_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);

        /* the "labels" field: its fixed length comes from the file */
        hid_t file_type = H5Dget_type(stars);
        hid_t label_type = H5Tget_member_type(
            file_type, H5Tget_member_index(file_type, "labels"));
        size_t label_len = H5Tget_size(label_type);
        hid_t labels_type = H5Tcreate(H5T_COMPOUND, label_len);
        H5Tinsert(labels_type, "labels", 0, label_type);
        char *labels = malloc(n * label_len);  /* not null terminated */
        H5Dread(stars, labels_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, labels);

        printf("star 0: x = %g kpc, label = %.*s\n",
               x[0], (int) label_len, labels);

        free(x);
        free(labels);
        H5Tclose(labels_type);
        H5Tclose(label_type);
        H5Tclose(file_type);
        H5Tclose(x_type);
        H5Sclose(space);
        H5Dclose(stars);
        H5Gclose(galaxy);
        H5Fclose(file);
        return 0;
    }


C++
~~~

With the HDF5 C++ API (compile with ``h5c++ read_galaxy.cpp``):

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <vector>

    #include <H5Cpp.h>

    int main() {
        H5::H5File file("decomposed.h5", H5F_ACC_RDONLY);

        double version;
        file.openAttribute("format_version")
            .read(H5::PredType::NATIVE_DOUBLE, &version);
        std::cout << "format version: " << version << "\n";

        H5::Group galaxy = file.openGroup("galaxy");
        H5::Attribute method_attr = galaxy.openAttribute("method");
        std::string method;
        method_attr.read(method_attr.getStrType(), method);
        std::cout << "method: " << method << "\n";

        H5::DataSet stars = galaxy.openDataSet("stars");
        hsize_t n = stars.getSpace().getSimpleExtentNpoints();

        // the "x" field alone: a compound type with only that member
        H5::CompType x_type(sizeof(double));
        x_type.insertMember("x", 0, H5::PredType::NATIVE_DOUBLE);
        std::vector<double> x(n);
        stars.read(x.data(), x_type);

        // the "labels" field: its fixed length comes from the file
        H5::CompType file_type = stars.getCompType();
        H5::StrType label_type =
            file_type.getMemberStrType(file_type.getMemberIndex("labels"));
        size_t label_len = label_type.getSize();
        H5::CompType labels_type(label_len);
        labels_type.insertMember("labels", 0, label_type);
        std::vector<char> labels(n * label_len);  // not null terminated
        stars.read(labels.data(), labels_type);

        std::string label0(labels.data(), label_len);
        label0.resize(label0.find('\0') == std::string::npos
                          ? label_len : label0.find('\0'));
        std::cout << "star 0: x = " << x[0] << " kpc, label = " << label0
                  << "\n";
        return 0;
    }


Fortran
~~~~~~~

With the HDF5 Fortran library (compile with ``h5fc read_galaxy.f90``).
Reading the variable-length string attributes from Fortran needs C
pointers, so this example reads only the numeric data:

.. code-block:: fortran

    program read_galaxy
        use hdf5
        use iso_c_binding
        implicit none

        integer(hid_t) :: file, galaxy, stars, space, attr, x_type
        integer(hsize_t) :: n
        integer(hsize_t), dimension(1) :: dims, maxdims
        integer(size_t) :: dsize
        real(c_double), target :: version(1)
        real(c_double), allocatable, target :: x(:)
        integer :: err

        call h5open_f(err)
        call h5fopen_f("decomposed.h5", H5F_ACC_RDONLY_F, file, err)

        ! format version: a double attribute of the root
        call h5aopen_f(file, "format_version", attr, err)
        call h5aread_f(attr, H5T_NATIVE_DOUBLE, c_loc(version), err)
        call h5aclose_f(attr, err)
        print *, "format version:", version(1)

        call h5gopen_f(file, "galaxy", galaxy, err)
        call h5dopen_f(galaxy, "stars", stars, err)
        call h5dget_space_f(stars, space, err)
        call h5sget_simple_extent_dims_f(space, dims, maxdims, err)
        n = dims(1)

        ! the "x" field alone: a compound type with only that member
        call h5tget_size_f(H5T_NATIVE_DOUBLE, dsize, err)
        call h5tcreate_f(H5T_COMPOUND_F, dsize, x_type, err)
        call h5tinsert_f(x_type, "x", 0_size_t, H5T_NATIVE_DOUBLE, err)
        allocate(x(n))
        call h5dread_f(stars, x_type, c_loc(x), err)
        print *, "star 1: x =", x(1), "kpc"

        deallocate(x)
        call h5tclose_f(x_type, err)
        call h5sclose_f(space, err)
        call h5dclose_f(stars, err)
        call h5gclose_f(galaxy, err)
        call h5fclose_f(file, err)
        call h5close_f(err)
    end program read_galaxy
