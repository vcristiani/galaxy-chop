[![Build Status](https://travis-ci.com/vcristiani/galaxy-chop.svg?branch=master)](https://travis-ci.com/vcristiani/galaxy-chop)

# Welcome to `galaxy-chop`

`galaxy-chop` is a package where you can find the most used methods for galaxy dynamical decomposition.

## Methods implemented
- Abadi+ 2003
- Cristiani+
- Obreja+ 2019
- Du+...

# Requirements

Libraries gfortran, gcc and others...

# Installation
You could find `galaxy-chop` at PyPI. The standar instalation via pip:

    $ pip install galaxy-chop

# Documentation

## Input data
 - Masses (in mass solar units) 
 - Positions (in kpc)
 - Velocities (in km/s)
 
 **Note:** You must have one file for each particle type, e.g. the file _stars.dat_ must contain 7 columns: m, x, y, z, vx, vy, vz. Then you must have other file for dark matter particles and other for gas (if you have).

## Output data

You will get the dinamical decomposition of the galaxy obtained with the method you had choose.

### Additionally you could get:
- The gravitational potencial of the particles
- The rotation matrix of the galaxy with respect some dynamical axis (e.i: aling with the angular momentum or some compornent of it)

# Quickstart


# Based on 

- paper 1

# License

MIT

# Authors
- Valeria Cristiani (e-mail: valeria.cristiani@unc.edu.ar)
- Ornela Marioni (e-mail: ornela.marioni@unc.edu.ar)
- Nelson Villagra (e-mail: ntvillagra@gmail.com)
- Antonela Taverna (e-mail: ataverna@unc.edu.ar)
- Bruno Sanchez (e-mail: bruno.sanchez@duke.edu)
- Rafael Pignata (e-mail: rafael.pignata@unc.edu.ar)
 
