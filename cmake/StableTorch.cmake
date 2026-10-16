option(KLIB_WITH_CUDA "Build the optional CUDA add kernel" ON)

# Build dependencies: Python 3.12+, nanobind and Torch 2.14+ headers.
find_package(Python 3.12 REQUIRED COMPONENTS
  Interpreter Development.Module Development.SABIModule
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c [=[
import json
from pathlib import Path
import torch

print(json.dumps({"root": str(Path(torch.__file__).parent), "version": torch.__version__}))
]=]
  OUTPUT_VARIABLE TORCH_BUILD_INFO
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
string(JSON TORCH_ROOT GET "${TORCH_BUILD_INFO}" root)
string(JSON TORCH_BUILD_VERSION GET "${TORCH_BUILD_INFO}" version)
if(TORCH_BUILD_VERSION VERSION_LESS "2.14")
  message(FATAL_ERROR "Building klib requires Torch >= 2.14; the runtime ABI target remains Torch 2.13")
endif()

# Core extension: CPython abi3 (3.12+) and the Torch 2.13 stable ABI.
# Link only torch_cpu so the wheel also loads with CPU-only Torch.
find_library(TORCH_CPU_LIBRARY torch_cpu
  PATHS "${TORCH_ROOT}/lib"
  NO_DEFAULT_PATH NO_CACHE REQUIRED
)
nanobind_add_module(_stable NB_STATIC STABLE_ABI NOMINSIZE klib/ops/stable/bindings.cc)
target_compile_features(_stable PRIVATE cxx_std_20)
target_include_directories(_stable PRIVATE "${TORCH_ROOT}/include" "${PROJECT_SOURCE_DIR}/compat")
target_link_libraries(_stable PRIVATE "${TORCH_CPU_LIBRARY}" ${CMAKE_DL_LIBS})
# Newer headers are allowed, but direct API calls must stay within Torch 2.13.
# bindings.cc resolves the Torch 2.14 Tensor conversion functions with dlsym.
target_compile_definitions(_stable PRIVATE
  TORCH_TARGET_VERSION=0x020d000000000000 # 2.13.0
  KLIB_BUILD_TORCH="${TORCH_BUILD_VERSION}"
)
target_compile_options(_stable PRIVATE -O3)
set_target_properties(_stable PROPERTIES
  INSTALL_RPATH "$ORIGIN/../torch/lib"
  INSTALL_RPATH_USE_LINK_PATH FALSE
)
install(TARGETS _stable LIBRARY DESTINATION klib)

# Optional CUDA plugin: loaded by bindings.cc only when CUDA add is called.
if(KLIB_WITH_CUDA)
  include("${CMAKE_CURRENT_LIST_DIR}/FindPipCUDAToolkit.cmake")
  if(NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_NVCC_EXECUTABLE)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
  # Set defaults before compiler detection, respecting -D and CUDAARCHS overrides.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 CACHE STRING "CUDA architectures")
  endif()
  enable_language(CUDA)

  add_library(_cuda_add SHARED klib/ops/stable/cuda_add.cu)
  set_target_properties(_cuda_add PROPERTIES
    PREFIX ""
    CUDA_RUNTIME_LIBRARY Static
  )
  install(TARGETS _cuda_add LIBRARY DESTINATION klib)
endif()
