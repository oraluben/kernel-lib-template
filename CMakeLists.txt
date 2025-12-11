cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(bitforge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_BUILD_TYPE Debug)
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

add_library(add_fb OBJECT klib/ops/add/add_kernel.cu)
set_target_properties(add_fb PROPERTIES CUDA_FATBIN_COMPILATION ON)

get_filename_component(ADD_ABS "klib/ops/add/add_kernel.cu" ABSOLUTE)
get_filename_component(ADD_DIR "${ADD_ABS}" DIRECTORY)

add_custom_command(
    OUTPUT ${ADD_DIR}/add_kernel.h
    COMMAND bin2c -c
            $<TARGET_OBJECTS:add_fb> > ${ADD_DIR}/add_kernel.h
    DEPENDS $<TARGET_OBJECTS:add_fb>
    COMMENT "Embedding fatbins into header file"
    VERBATIM
)

add_library(add_ffi SHARED klib/ops/add/add_kernel.cc ${ADD_DIR}/add_kernel.h)
target_link_libraries(add_ffi PRIVATE tvm_ffi_shared cuda)
set_target_properties(add_ffi PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS add_ffi LIBRARY DESTINATION klib/ops)

add_library(alloc_ffi SHARED klib/ops/alloc/alloc_kernel.cc)
target_link_libraries(alloc_ffi PRIVATE tvm_ffi_shared cuda)
set_target_properties(alloc_ffi PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS alloc_ffi LIBRARY DESTINATION klib/ops)
