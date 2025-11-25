cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(bitforge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_RUNTIME_LIBRARY None)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
   set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_FOUND}")
   set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_FOUND}")
   set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_FOUND}")
endif()

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
find_package(tvm_ffi CONFIG REQUIRED)

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(USE_SABI USE_SABI ${SKBUILD_SABI_VERSION})
endif()

include_directories(klib/ops/include)

add_library(add_ffi SHARED klib/ops/add/add_kernel.cu)
target_link_libraries(add_ffi PRIVATE tvm_ffi_shared)
set_target_properties(add_ffi PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS add_ffi LIBRARY DESTINATION klib/ops)
