cmake_minimum_required(VERSION 3.18)

project(galaxychop
  VERSION 0.3.0
  DESCRIPTION "Potential calculator"
  LANGUAGES C Fortran
)

if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
endif()

# Safety net
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
  )
endif()

find_package(PythonExtensions)
find_package(OpenMP)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# F2PY headers
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# =============================================================================
# Compíling potential module
message(STATUS "Compiling fortran potential...")
set(f2py_module_name "potential")
set(f2py_module_dir "galaxychop/preproc/fortran")
set(
    fortran_src_file 
    "${CMAKE_SOURCE_DIR}/galaxychop/preproc/fortran/potential.f90"
)
set(
    generated_module_file 
    "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_dir}/${f2py_module_name}${PYTHON_EXTENSION_MODULE_SUFFIX}"
)
message(STATUS "Module_file: ${generated_module_file}")
add_custom_target(${f2py_module_name}_mod ALL DEPENDS ${generated_module_file})
add_custom_command(
  OUTPUT ${generated_module_file}
  COMMAND ${PYTHON_EXECUTABLE}  -m "numpy.f2py"
    -c
    ${fortran_src_file}
    -m ${f2py_module_name}
    --f90flags='-fopenmp' -lgomp
  WORKING_DIRECTORY ${f2py_module_dir}
  DEPENDS ${fortran_src_file}
)
install(
  FILES ${generated_module_file} 
  DESTINATION ${f2py_module_dir}
)
# =============================================================================
