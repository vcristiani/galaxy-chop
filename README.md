[![Build Status](https://travis-ci.com/vcristiani/galaxy-chop.svg?branch=master)](https://travis-ci.com/vcristiani/galaxy-chop)
[![Documentation Status](https://readthedocs.org/projects/galaxy-chop/badge/?version=latest)](https://galaxy-chop.readthedocs.io/en/latest/?badge=latest)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)

![logo](https://github.com/vcristiani/galaxy-chop/blob/master/docs/galaxychop_logo.png)

# Welcome to **GalaxyChop**

**GalaxyChop**  is a Python package that tackles the dynamical decomposition problem by utilizing clustering techniques in phase space for stellar galactic components.

It runs in numerical N-body simulations populated with Semi-analytical models and full Hydrodynamical simulations, such as Illustris TNG.

## Dynamic decomposition method implemented
- **[GCAbadi](https://galaxy-chop.readthedocs.io/en/latest/api/galaxychop.html#galaxychop.models.GCAbadi):** Implementation of the method for dynamically decomposing galaxies described by [Abadi et al.(2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...597...21Aabstract). 
- **[GCChop](https://galaxy-chop.readthedocs.io/en/latest/api/galaxychop.html#galaxychop.models.GCChop):** Implementation of the method for dynamically decomposing galaxies used in [Tissera et al.(2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.420..255T/abstract), [Vogelsberger et al.(2014)](https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V/abstract), [Marinacci et al.(2014)](https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1750M/abstract), [Park et al.(2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...883...25P/abstract), etc.
- **[GCKmeans](https://galaxy-chop.readthedocs.io/en/latest/api/galaxychop.html#galaxychop.models.GCKmeans):** Implementation of [Skitlearn](https://scikit-learn.org/stable/about.html#citing-scikit-learn) K-means as a method for dynamically decomposing galaxies. 
- **[GCgmm](https://galaxy-chop.readthedocs.io/en/latest/api/galaxychop.html#galaxychop.models.GCGmm):** Implementation of the method for dynamically decomposing galaxies described by [Obreja et al.(2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.4915O/abstract).
- **[GCAutogmm](https://galaxy-chop.readthedocs.io/en/latest/api/galaxychop.html#galaxychop.models.GCAutogmm):** Implementation of the method for dynamically decomposing galaxies described by [Du et al.(2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...884..129D/abstract)

## Requirements

You need Python 3.7 or 3.8 to run GalaxyChop.

## Standard Installation

You could find **GalaxyChop**  at PyPI. The standar instalation via pip:

    $ pip install galaxychop

## Development Install

Clone this repo and then inside the local directory execute

     $ git clone https://github.com/vcristiani/galaxy-chop.git
     $ cd galaxy-chop
     $ pip install -e .

## Helpful links
- [GalaxyChop's Tutorial](https://galaxy-chop.readthedocs.io/en/latest/tutorial.html)
- [Licence](https://galaxy-chop.readthedocs.io/en/latest/license.html)
- [GalaxyChop API](https://galaxy-chop.readthedocs.io/en/latest/api/galaxychop.html)


## Authors
- Valeria Cristiani (e-mail: valeria.cristiani@unc.edu.ar)
- Ornela Marioni (e-mail: ornela.marioni@unc.edu.ar)
- Nelson Villagra (e-mail: ntvillagra@gmail.com)
- Antonela Taverna (e-mail: ataverna@unc.edu.ar)
- Bruno Sanchez (e-mail: bruno.sanchez@duke.edu)
- Rafael Pignata (e-mail: rafael.pignata@unc.edu.ar)

