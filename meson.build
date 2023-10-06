project('galaxychop', 'c',
  version : '0.3.dev0',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print("../" + os.path.relpath(numpy.get_include()))'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print("../" + os.path.relpath(numpy.f2py.get_include()))'],
    check : true
).stdout().strip()

omp = dependency('openmp')

inc_np = include_directories(incdir_numpy, incdir_f2py)

potential_source = custom_target('potentialmodule.c',
  input : ['galaxychop/preproc/fortran/potential.f90'],
  output : [
    'potentialmodule.c', 
    'potential-f2pywrappers2.f90'
    ],
  command : [
    py, '-m', 'numpy.f2py', '@INPUT@', 
    '-m', 'potential', '--lower'
  ]
)

f_sources = ['galaxychop/preproc/fortran/__init__.py']
py.install_sources(f_sources, subdir:'galaxychop/preproc/fortran')

main_sources = [
  'galaxychop/constants.py',
  'galaxychop/io.py',
  'galaxychop/__init__.py',
]
py.install_sources(main_sources, subdir: 'galaxychop')
  

core_sources = [
  'galaxychop/core/sdynamics.py',
  'galaxychop/core/__init__.py',
  'galaxychop/core/plot.py',
  'galaxychop/core/data.py',
]
py.install_sources(core_sources, subdir:'galaxychop/core')

models_sources = [ 
  'galaxychop/models/__init__.py',
  'galaxychop/models/threshold.py',
  'galaxychop/models/histogram.py',
  'galaxychop/models/gaussian_mixture.py',
  'galaxychop/models/kmeans.py',
  'galaxychop/models/_base.py',
]
py.install_sources(models_sources, subdir:'galaxychop/models')

preproc_sources = [
  'galaxychop/preproc/pcenter.py',
  'galaxychop/preproc/__init__.py',
  'galaxychop/preproc/potential_energy.py',
  'galaxychop/preproc/salign.py',
  'galaxychop/preproc/smr_crop.py',
]

py.install_sources(preproc_sources, subdir:'galaxychop/preproc')

utils_sources = [
  'galaxychop/utils/__init__.py',
  'galaxychop/utils/decorators.py'
]
py.install_sources(models_sources, subdir:'galaxychop/models')

preproc_sources = [
  'galaxychop/preproc/pcenter.py',
  'galaxychop/preproc/__init__.py',
  'galaxychop/preproc/potential_energy.py',
  'galaxychop/preproc/salign.py',
  'galaxychop/preproc/smr_crop.py',
]
py.install_sources(preproc_sources, subdir:'galaxychop/preproc')
  
f_sources = ['galaxychop/preproc/fortran/__init__.py']
py.install_sources(f_sources, subdir:'galaxychop/preproc/fortran')

utils_sources = [
  'galaxychop/utils/__init__.py',
  'galaxychop/utils/decorators.py'
]
py.install_sources(utils_sources, subdir:'galaxychop/utils')

py.extension_module('potential',
  ['galaxychop/preproc/fortran/potential.f90', potential_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : [py_dep, omp],
  c_args : '-fopenmp',
  link_args : '-lgomp',
  install : true,
  subdir: 'galaxychop/preproc/fortran/'
)
