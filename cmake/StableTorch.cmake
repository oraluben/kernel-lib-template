find_package(Python 3.12 REQUIRED COMPONENTS
  Interpreter Development.Module Development.SABIModule
)
find_package(nanobind CONFIG REQUIRED)

# TorchConfig manages CUDA code generation through TORCH_CUDA_ARCH_LIST.
if(NOT DEFINED TORCH_CUDA_ARCH_LIST AND "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
  set(TORCH_CUDA_ARCH_LIST "8.0;8.6;8.9;9.0+PTX" CACHE STRING "CUDA architectures")
endif()
find_package(Torch 2.14 REQUIRED)

nanobind_add_module(_stable NB_STATIC STABLE_ABI NOMINSIZE
  klib/ops/stable/bindings.cc
  klib/ops/add/add_kernel.cu
)
target_compile_features(_stable PRIVATE cxx_std_20)
target_include_directories(_stable PRIVATE ${TORCH_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/compat")
target_link_libraries(_stable PRIVATE "$<TARGET_FILE:torch_cpu>" ${CMAKE_DL_LIBS})
# Build with 2.14+ headers, but keep direct Torch calls within the 2.13 ABI.
# bindings.cc resolves the 2.14 Tensor conversion functions with dlsym.
target_compile_definitions(_stable PRIVATE
  TORCH_TARGET_VERSION=0x020d000000000000 # 2.13.0
  KLIB_BUILD_TORCH="${Torch_VERSION}"
)
target_compile_options(_stable PRIVATE -O3)
set_target_properties(_stable PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
  INSTALL_RPATH "$ORIGIN/../torch/lib"
  INSTALL_RPATH_USE_LINK_PATH FALSE
)
install(TARGETS _stable LIBRARY DESTINATION klib)
