cmake_minimum_required(VERSION 3.17.0)

project(galaxychop C)

option(SKBUILD "Should be ON of being build by skbuild, 
and OFF of being build by regular cmake" OFF)

if (NOT SKBUILD)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake/")
endif()

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

find_package(PythonExtensions REQUIRED)
find_package(NumPy)
find_package(F2PY REQUIRED)

message(STATUS "F2PY included from: ${F2PY_INCLUDE_DIR}")

# =============================================================================
# Potential module
# =============================================================================
message(STATUS "Compiling fortran potential backend...")
set(f2py_module_name "potential")
set(f2py_module_dir "galaxychop/preproc/fortran")
set(
    fortran_src_file 
    "${CMAKE_SOURCE_DIR}/galaxychop/preproc/fortran/potential.f90"
)
set(
    generated_module_file 
    ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_name}${PYTHON_EXTENSION_MODULE_SUFFIX}
)

add_custom_target(${f2py_module_name} ALL
  DEPENDS ${generated_module_file}
)

add_custom_command(
  OUTPUT ${generated_module_file}
  COMMAND ${F2PY_EXECUTABLE}
    -m ${f2py_module_name}
    -c
    ${fortran_src_file}
    --f90flags="-fopenmp" -lgomp
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${fortran_src_file}
)
# =============================================================================

if (SKBUILD)
    install(FILES ${generated_module_file} DESTINATION ${f2py_module_dir})
else()
  install(FILES ${generated_module_file} DESTINATION ${f2py_module_dir})
endif()
