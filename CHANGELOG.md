# GalaxyChop Changelog


<!-- BODY -->

## Version 0.3

- Now the decomposition models are stateless and return a `Component`
  object that can be used as a hue in all plots, and can calculate mass
  fractions for each component
- Components and plots now support a `lmap` parameter (label-map)
  which allows to arbitrarily change component names.
  In addition, the models that "know" which component is which automatically
  assign the lmaps.
- A persistence based on hdf5 was implemented.
- All preprocessing utilities now live in the `preproc` package.
- The `utils` package now only has modules useful for GalaxyChop development.
- Migrated the entire galaxy architecture to a `ParticleSet` that abstracts
  over the concept of Gas, DM and Stars independently but with the same code.
- The quality has been tightened.


## Version 0.2

- First public release