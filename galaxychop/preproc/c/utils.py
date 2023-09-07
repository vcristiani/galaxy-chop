import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_float
import os

def check(exe, recompilar=True):

  if not os.path.exists('%s.so' % exe) or recompilar:
    print("Compile %s.c" % exe)
    comando = "gcc -c -O3 -fPIC -Wall -lm -fopenmp %s.c -o %s.o" % (exe, exe)
    os.system(comando)
    if os.path.exists('%s.o' % exe):
      print("Compile %s.so" % exe)
      comando = "gcc -shared -Wall -lm -fopenmp %s.o -o %s.so" % (exe, exe)
      os.system(comando)
    else:
      print("Error!!!\n")
      exit(0)

  return

def calcula_potencial(Npart, mp, x, y, z):

  array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
  array_1d_int   = npct.ndpointer(dtype=np.int32,   ndim=1, flags='C_CONTIGUOUS')
 
  libcd = npct.load_library("potencial.so", ".")
 
  libcd.calculate_potential.restype = None
  libcd.calculate_potential.argtypes = [ \
  c_int, array_1d_float, array_1d_float, array_1d_float, array_1d_float, array_1d_float]

  Ep = np.zeros(Npart, dtype=np.float32)

  libcd.calculate_potential(Npart, mp, x, y, z, Ep)

  return Ep 
