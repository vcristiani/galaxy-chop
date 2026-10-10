# Welcome to **GalaxyChop**

![logo](https://github.com/vcristiani/galaxy-chop/raw/master/docs/source/_static/galaxychop_logo_wb.png)

<!-- BODY -->

[![GalaxyChop CI](https://github.com/vcristiani/galaxy-chop/actions/workflows/CI.yml/badge.svg)](https://github.com/vcristiani/galaxy-chop/actions/workflows/CI.yml)
[![Documentation Status](https://readthedocs.org/projects/galaxy-chop/badge/?version=latest)](https://galaxy-chop.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/galaxychop)](https://pypi.org/project/galaxychop/)
[![License](https://img.shields.io/pypi/l/galaxychop?color=blue)](https://raw.githubusercontent.com/vcristiani/galaxy-chop/master/LICENSE.txt)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/galaxychop/)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)

**GalaxyChop** is a Python package that tackles the dynamical decomposition problem by using clustering techniques in phase space for stellar galactic components.

It runs in numerical N-body simulations populated with semi-analytical models and full hydrodynamical simulations, such as [Illustris TNG](https://www.tng-project.org/) and [EAGLE](http://icc.dur.ac.uk/Eagle/).

## 🌌 Motivation

Galaxies are self-gravitating complex stellar systems formed mainly by stars, dark matter, gas and dust. Stars are assembled in different stellar components, such as the disk (thin and thick), the nucleus, the stellar halo and the bar. The components interact with each other and each of them follows its own temporal evolution. For this reason, the description of the formation and evolution of galaxies is strongly linked to the formation and evolution of each of these individual components and their assembly in the final galaxy.

Dynamical decomposition is a fundamental tool to separate each galaxy component for further study. Numerous methods exist in the literature to perform this task, but there is no tool that allows us to use several of them, providing the possibility of an easy comparison.

## 🧩 Dynamic decomposition methods implemented

- **JHistogram:** Implementation of the dynamic decomposition model of galaxies described by [Abadi et al. (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...597...21Aabstract).
- **JThreshold:** Implementation of the dynamic decomposition model of galaxies used in [Tissera et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.420..255T/abstract), [Vogelsberger et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V/abstract), [Marinacci et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1750M/abstract), [Park et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...883...25P/abstract), etc.
- **KMeans:** Implementation of [Scikit-Learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn) K-means as a model for dynamical decomposition of galaxies.
- **GaussianMixture:** Implementation of the dynamic decomposition model of galaxies described by [Obreja et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.4915O/abstract).
- **AutoGaussianMixture:** Implementation of the dynamic decomposition model of galaxies described by [Du et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...884..129D/abstract).
- **JEHistogram:** Implementation of the dynamic decomposition model of galaxies described by [Cristiani et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...692A..63C/abstract).

**And many more.**

## 🔧 Requirements

You need Python `>= 3.11` to run GalaxyChop.

### Standard Installation

You can find **GalaxyChop** on PyPI. The standard installation via pip:

```bash
$ pip install galaxychop
```

### Development Install

Clone this repo and then, inside the local directory, execute

```bash
$ git clone https://github.com/vcristiani/galaxy-chop.git
$ cd galaxy-chop
$ pip install -r requirements_dev.txt
```

## 💻 Command Line

Installing GalaxyChop also installs the `galaxychop` command:

```bash
$ galaxychop methods                       # list the decomposition methods
$ galaxychop info decomposed.h5            # total mass of each type/component
$ galaxychop decompose galaxy.h5 decomposed.h5 --method JHistogram
```

`decompose` centers and aligns the galaxy before decomposing it (use
`--no-align` to skip it) and stores the result in a new HDF5 file. Run
`galaxychop --help` or `galaxychop COMMAND --help` for every option.

## 📦 Code Repository & Issues

<https://github.com/vcristiani/galaxy-chop>

## 📜 License

GalaxyChop is under [The MIT License](https://raw.githubusercontent.com/vcristiani/galaxy-chop/master/LICENSE.txt)

This license allows unlimited redistribution for any purpose as long as its copyright notices and the license's disclaimers of warranty are maintained.

## 📚 Citation

If you use GalaxyChop in a scientific publication, we would appreciate citations to the following paper:

> Cristiani, V. A., Abadi, M. G., Taverna, A., Cabral, J., Benelli, F., & Sánchez, B. (2024). Untangling stellar components of galaxies: Evaluation of dynamical decomposition methods in simulated galaxies with GalaxyChop. *Astronomy & Astrophysics*, 692, A63.

Bibtex entry:

```bibtex
@article{cristiani2024untangling,
  title={Untangling stellar components of galaxies: Evaluation of dynamical decomposition methods in simulated galaxies with GalaxyChop},
  author={Cristiani, Valeria A and Abadi, Mario G and Taverna, Antonela and Cabral, Juan and Benelli, Federico and S{\'a}nchez, Bruno},
  journal={Astronomy \& Astrophysics},
  volume={692},
  pages={A63},
  year={2024},
  publisher={EDP Sciences}
}
```

**Full texts:** 
- <https://www.aanda.org/articles/aa/pdf/2024/12/aa51202-24.pdf>
- <https://arxiv.org/pdf/2410.00105>

## 👥 Authors

- Valeria Cristiani [valeria.cristiani@unc.edu.ar](mailto:valeria.cristiani@unc.edu.ar) ([IATE-OAC-CONICET][], [FaMAF-UNC][]).
- Antonela Taverna ([IATE-OAC-CONICET][]).
- Juan Cabral ([IATE-OAC-CONICET][], [CONAE][]).
- Federico Benelli ([IPQA-CONICET][], [FCEFyN-UNC][]).
- Bruno Sanchez ([Duke University][]).
- Bruno Celiz [bruno.celiz@mi.unc.edu.ar](mailto:bruno.celiz@mi.unc.edu.ar) ([IATE-OAC-CONICET][], [FaMAF-UNC][]).
- Daniela Stauber ([FaMAF-UNC][]).

  [IATE-OAC-CONICET]: http://iate.oac.uncor.edu/
  [OAC-CONICET]: https://oac.unc.edu.ar/
  [FaMAF-UNC]: https://www.famaf.unc.edu.ar/
  [CONAE]: https://www.argentina.gob.ar/ciencia/conae
  [Duke University]: https://duke.edu/
  [IPQA-CONICET]: https://ipqa.unc.edu.ar/en/
  [FCEFyN-UNC]: https://fcefyn.unc.edu.ar/
