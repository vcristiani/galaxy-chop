import galaxychop as gchop
import numpy as np
#gal = gchop.read_hdf5("/home/juanbc/Descargas/galaxy_TNG_17234.h5")
# gal = gchop.read_hdf5("/home/juanbc/Descargas/galaxy_TNG_20.h5")
# gal = gchop.read_hdf5("/home/juanbc/Descargas/galaxy_TNG_60737.h5")
gal = gchop.read_hdf5("tests/datasets/gal394242.h5")
gal
gal = gchop.preproc.center_and_align(gal)
id(gal.stars), id(gal.stars.copy())
#comps = gchop.models.JHistogram().decompose(gal)
comps = gchop.models.AutoGaussianMixture().decompose(gal)
df = comps.to_dataframe( attributes=["ptype", "x", "label"])
import ipdb; ipdb.set_trace()