[![Build Status](https://travis-ci.com/vcristiani/galaxy-chop.svg?branch=master)](https://travis-ci.com/vcristiani/galaxy-chop)
[![Documentation Status](https://readthedocs.org/projects/galaxy-chop/badge/?version=master)](https://galaxy-chop.readthedocs.io/en/master/?badge=master)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)

![logo](https://github.com/vcristiani/galaxy-chop/blob/master/docs/galaxychop_logo.png)

# Welcome to **GalaxyChop**

**GalaxyChop**  is a Python code that tackles the dynamical decomposition problem by utilizing clustering techniques in phase space for stellar galactic components.

It runs in numerical N-body simulations populated with Semi-analytical models and full Hydrodynamical simulations, such as Illustris TNG.

## Dynamic decomposition models implemented
- **GCAbadi:** Implementation of the method for dynamically decomposing galaxies described by [Abadi et al.(2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...597...21Aabstract). 
- **GCChop:** Implementation of the method for dynamically decomposing galaxies used in [Tissera et al.(2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.420..255T/abstract), [Vogelsberger et al.(2014)](https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V/abstract), [Marinacci et al.(2014)](https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1750M/abstract), [Park et al.(2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...883...25P/abstract), etc.
- **GCgmm:** Implementation of the method for dynamically decomposing galaxies described by [Obreja et al.(2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.4424O/abstract).
- **GCAutogmm:** Implementation of the method for dynamically decomposing galaxies described by [Du et al.(2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...884..129D/abstract)
- **GCKmeans:** Implementation of [Skitlearn](https://scikit-learn.org/stable/about.html#citing-scikit-learn) K-means. 

## Requirements

You need Python 3.7 or 3.8 to run GalaxyChop.

## Standard Installation

You could find **GalaxyChop**  at PyPI. The standar instalation via pip:

    $ pip install galaxy-chop

## Development Install

Clone this repo and then inside the local directory execute

     $ pip install -e .

# Authors
- Valeria Cristiani (e-mail: valeria.cristiani@unc.edu.ar)
- Ornela Marioni (e-mail: ornela.marioni@unc.edu.ar)
- Nelson Villagra (e-mail: ntvillagra@gmail.com)
- Antonela Taverna (e-mail: ataverna@unc.edu.ar)
- Bruno Sanchez (e-mail: bruno.sanchez@duke.edu)
- Rafael Pignata (e-mail: rafael.pignata@unc.edu.ar)

