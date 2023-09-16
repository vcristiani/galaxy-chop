# GalaxyChop Changelog


<!-- BODY -->

## Version 0.3

- Every implemented method is a-priori stable; for future changes, a deprecation strategy will be implemented.

- Multiple utilities to transform the galaxy in multiple formats:

    - `Galaxy.to_dict()`: This method converts the Galaxy object into a Python dictionary. It extracts all the relevant attributes and their values from the Galaxy object and organizes them into a dictionary format and by coercing all the attributes units.
    - `Galaxy.disassemble()`: Used to break down a complex Galaxy object into its individual components or sub-elements in a signle plain dictionary. The output of this method can be used to create a new galaxy with the `galaxychop.mkgalaxy()` function.
    - `Galaxy.to_dataframe()`: Responsible for converting a Galaxy object into a pandas DataFrame. This is particularly useful when you want to perform data analysis or manipulation using the powerful features of pandas.
    Galaxy.to_hdf5():
    - `Galaxy.to_hdf5()` method is used to save the Galaxy data in the [HDF5 (Hierarchical Data Format version 5)](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file format. HDF5 is a versatile and efficient file format for storing large and complex datasets. This method allows you to serialize the Galaxy object and store it in an HDF5 file, making it accessible for later retrieval and analysis.

- The utility previously implemented in the jcirc function has now become a method
  within the `Galaxy` class called `Galaxy.stellar_dynamics()`.

- Now the decomposition models are stateless and return a `Component`
  object that can be used as a hue in all plots, and can calculate mass
  fractions for each component.

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