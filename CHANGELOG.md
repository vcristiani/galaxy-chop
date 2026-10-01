# GalaxyChop Changelog


<!-- BODY -->

## Version 0.3

- Every implemented method is a-priori stable; for future changes, a deprecation strategy will be implemented.

- Multiple utilities to transform the galaxy in multiple formats:

    - `Galaxy.to_dict()`: This method converts the Galaxy object into a Python dictionary. It extracts all the relevant attributes and their values from the Galaxy object and organizes them into a dictionary format and by coercing all the attributes units.
    - `Galaxy.disassemble()`: Used to break down a complex Galaxy object into its individual components or sub-elements in a signle plain dictionary. The output of this method can be used to create a new galaxy with the `galaxychop.mkgalaxy()` function.
    - `Galaxy.to_dataframe()`: Responsible for converting a Galaxy object into a pandas DataFrame. This is particularly useful when you want to perform data analysis or manipulation using the powerful features of pandas.
    - `Galaxy.to_hdf5()`: This method is used to save the Galaxy data in the [HDF5 (Hierarchical Data Format version 5)](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file format. HDF5 is a versatile and efficient file format for storing large and complex datasets. This method allows you to serialize the Galaxy object and store it in an HDF5 file, making it accessible for later retrieval and analysis.

- The utility previously implemented in the jcirc function has now become a method
  within the `Galaxy` class called `Galaxy.stellar_dynamics()`.

- Now the decomposition models are stateless and return a `DecomposedGalaxy`
  object (formed by `DecomposedParticleSet` instances) that can be used as a
  hue in all plots, and can calculate deterministic and probabilistic mass
  fractions for each component via `DecomposedGalaxy.total_mass()`.

- Decompositions now track whether they are deterministic (hard assignment,
  e.g. KMeans) or probabilistic (soft/fuzzy assignment, e.g. GaussianMixture)
  through the `has_probabilities` flag, consistently propagated across
  `DecomposedParticleSet`, `DecomposedGalaxy` and HDF5 persistence. For
  probabilistic decompositions, `total_mass()` additionally reports the
  expected ("probabilistic") mass and mass fraction per component, weighting
  each particle's mass by its membership probability.

- `Galaxy.to_hdf5()` / `galaxychop.read_hdf5()` now also support
  `DecomposedGalaxy` objects, persisting the decomposition method, the
  component/label assignment and, when present, the membership
  probabilities. Reading older HDF5 files (format versions 1.0 and 2.0)
  remains supported for backwards compatibility.

- Fixed a bug in the common decomposition pipeline
  (`GalaxyDecomposerABC.decompose()`) that could silently assign particles
  to the wrong component: the per-particle component array was sorted by
  value before being written back into the full particle array, which only
  produced correct results when a decomposer happened to return its labels
  already in ascending order. This affected every decomposer that goes
  through `decompose()` (JThreshold, JHistogram, KMeans, GaussianMixture,
  AutoGaussianMixture, JEHistogram).

- All parameters with defaults, now are keyword only.

- Components and plots now support a `lmap` parameter (label-map)
  which allows to arbitrarily change component names.
  In addition, the models that "know" which component is which automatically
  assign the lmaps.

- All preprocessing utilities now live in the `preproc` package.

  It was also unified the idea that it is a 'preprocessor,' something that takes in a galaxy and returns a transformed galaxy. Additionally, we accept some functions in this package that serve to evaluate the transformations.

- The `utils` package now only has modules useful for GalaxyChop development.

- Migrated the entire galaxy architecture to a `ParticleSet` that abstracts
  over the concept of Gas, DM and Stars independently but with the same code.

- The quality has been tightened.


## Version 0.2

- First public release