# FindPipCUDAToolkit.cmake
#
# Locate CUDA toolkit installed via pip packages:
#   pip install nvidia-cuda-nvcc nvidia-cuda-cccl
#
# This module should be included BEFORE project() to set CMAKE_CUDA_COMPILER.
# It uses the Python interpreter to locate the pip-installed nvidia packages.
#
# The module sets the following variables when pip CUDA is found:
#   CMAKE_CUDA_COMPILER  - path to nvcc
#   CUDAToolkit_ROOT     - root of the CUDA toolkit
#   CUDAToolkit_INCLUDE_DIRS - include directories (toolkit + cccl)

if(CMAKE_CUDA_COMPILER OR DEFINED ENV{CUDACXX} OR EXISTS "/usr/local/cuda/bin/nvcc")
  return()
endif()

find_program(_PIP_CUDA_PYTHON_EXE NAMES python3 python)

if(NOT _PIP_CUDA_PYTHON_EXE)
  return()
endif()

execute_process(
  COMMAND "${_PIP_CUDA_PYTHON_EXE}" "${CMAKE_CURRENT_LIST_DIR}/find_pip_cuda.py"
  OUTPUT_VARIABLE _PIP_CUDA_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _PIP_CUDA_RESULT
)

if(NOT _PIP_CUDA_RESULT EQUAL 0)
  message(STATUS "FindPipCUDAToolkit: pip-installed CUDA toolkit not found")
  return()
endif()

string(JSON _PIP_CUDA_NVCC GET "${_PIP_CUDA_OUTPUT}" "nvcc")
string(JSON _PIP_CUDA_ROOT GET "${_PIP_CUDA_OUTPUT}" "root")
string(JSON _PIP_CUDA_INCLUDE GET "${_PIP_CUDA_OUTPUT}" "include")

set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_NVCC}" CACHE FILEPATH "CUDA compiler (from pip)" FORCE)
set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}" CACHE PATH "CUDA toolkit root (from pip)" FORCE)
set(CUDAToolkit_INCLUDE_DIRS "${_PIP_CUDA_INCLUDE}" CACHE PATH "CUDA toolkit include (from pip)")

# Add library paths so the linker can find -lcuda and other CUDA libs
list(APPEND CMAKE_LIBRARY_PATH "${_PIP_CUDA_ROOT}/lib/stubs" "${_PIP_CUDA_ROOT}/lib")
set(_PIP_CUDA_STUBS_DIR "${_PIP_CUDA_ROOT}/lib/stubs" CACHE PATH "" FORCE)

string(JSON _PIP_CUDA_CCCL_INCLUDE ERROR_VARIABLE _err GET "${_PIP_CUDA_OUTPUT}" "cccl_include")
if(NOT _err)
  list(APPEND CUDAToolkit_INCLUDE_DIRS "${_PIP_CUDA_CCCL_INCLUDE}")
endif()

message(STATUS "FindPipCUDAToolkit: using pip-installed CUDA toolkit")
message(STATUS "  nvcc: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  root: ${CUDAToolkit_ROOT}")
