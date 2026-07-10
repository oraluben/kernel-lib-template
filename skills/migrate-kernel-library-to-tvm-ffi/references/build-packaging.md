# CMake, Scikit-Build-Core, Stubs, And Stable API

## Contents

- Migration boundary
- Pyproject pattern
- CMake pattern
- Source and post-build behavior
- CUDA stubs
- Python Stable ABI
- Native ABI and RPATH

## Migration Boundary

Inventory all `setup.py` behavior before deleting or bypassing it:

- version generation and local metadata;
- source manifests and generated instantiations;
- CUDA architectures and compiler flags;
- include and library discovery;
- custom build extensions;
- SASS patching, spill checks, and binary rewriting;
- generated Python modules and `.pyi` files;
- symlink/copy behavior and runtime JIT headers;
- cached-wheel commands and environment variables.

Move custom logic to focused scripts or CMake custom commands. Do not silently drop it.

## Pyproject Pattern

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "apache-tvm-ffi>=0.1.8"]
build-backend = "scikit_build_core.build"

[project]
name = "my-kernel-lib"
dynamic = ["version"]
dependencies = ["apache-tvm-ffi", "torch>=2.4"]

[tool.scikit-build]
wheel.packages = ["my_kernel_lib"]
cmake.version = ">=3.27"
build-dir = "build/{wheel_tag}"
wheel.py-api = "cp38"
```

Keep Torch in runtime dependencies if the Python API returns Torch tensors or uses Torch callbacks. Removing libtorch linkage is distinct from removing Python Torch.

Use isolated builds only when build-system requirements include every needed toolchain package. For a pip CUDA toolchain already installed in the active environment, document `--no-build-isolation` or add matching packages to build requirements/options.

Large CUDA libraries have two independent concurrency multipliers: the number of build jobs and nvcc's per-process `--threads` value. Bound both explicitly on high-core-count builders:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2 \
  python -m pip wheel . --no-build-isolation --no-deps
```

Choose limits from the worker's memory budget. Do not let Ninja use hundreds of visible CPUs while every nvcc process also starts dozens of threads; this can OOM an otherwise large machine. Keep the build directory and ccache reusable so a lower-concurrency retry resumes from completed objects.

## CMake Pattern

Discover tvm-ffi from the build interpreter:

```cmake
find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_VARIABLE TVM_FFI_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TVM_FFI_CMAKE_DIR}")
find_package(tvm_ffi CONFIG REQUIRED)
```

Build a regular library for `tvm_ffi.load_module`:

```cmake
add_library(my_ops SHARED ${HOST_SOURCES} ${CUDA_SOURCES})
target_link_libraries(my_ops PRIVATE tvm_ffi::shared CUDA::cudart)
set_target_properties(my_ops PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS my_ops LIBRARY DESTINATION my_kernel_lib/ops)
```

Use `Python_add_library` only when the target actually uses CPython's extension-module ABI. A tvm-ffi library loaded with `dlopen` does not need a CPython module init symbol or Python development headers.

Preserve separable compilation, device linking, architecture-specific `-gencode` behavior, per-language options, and linker options. Generator expressions are preferable to global `CMAKE_CXX_FLAGS` mutation.

## Source And Post-Build Behavior

- Prefer explicit checked-in source lists for stable kernels.
- For generated instantiation directories, a scoped `file(GLOB CONFIGURE_DEPENDS ...)` is acceptable when every matching file must be compiled.
- Avoid globbing an entire vendored CUTLASS tree.
- Express SASS patching or spill checks as custom commands/targets with declared inputs, outputs, and tool dependencies.
- Make optional developer checks separately controllable from required wheel transformations.
- Install runtime JIT headers and generated Python files exactly once.

## CUDA Stubs

Direct linking is the default. Stubs are justified only when all of these hold:

1. The build/repair environment lacks a runtime library that must remain external.
2. The project has a documented runtime loader for the real library.
3. The stub covers every referenced symbol or unresolved symbols are intentionally deferred.
4. `ldd -r`, `nm -u`, and import tests are run without relying on Torch to preload missing libraries.

CUDA driver stubs are common for link-time `libcuda.so`; broad hand-written cudart/NVRTC/cuBLASLt forwarding layers are high maintenance. Prefer normal shared-library dependencies plus controlled auditwheel exclusions and RPATH when deployment guarantees them.

On a CPU-only validation host, the toolkit may provide only `stubs/libcuda.so` while the binary requests the `libcuda.so.1` SONAME. A temporary `libcuda.so.1` symlink in a test-only directory can make `ldd -r` and module registration checks possible. Never package that symlink or claim it can execute kernels; production must resolve the real NVIDIA driver library.

After adding or removing stubs, verify:

- dynamic `NEEDED` entries;
- symbol versions;
- loader search order and RPATH/RUNPATH;
- behavior in a process that has not imported Torch first;
- failure messages when the real library is absent.

## Python Stable ABI

Separate three claims:

1. **Public Python API stability**: Python function names/signatures remain compatible.
2. **CPython Stable ABI (`abi3`)**: native Python extension symbols use the limited API for a declared minimum Python version.
3. **Native C++/CUDA ABI**: shared-library and compiler/runtime dependencies remain compatible.

For a regular tvm-ffi-loaded `.so`, the library normally has no CPython dependency. The Python wheel still depends on the `tvm_ffi` package, whose own extension handles Python integration. `wheel.py-api = "cp38"` can produce a `cp38-abi3` wheel tag, but the tag is a distribution promise, not proof.

Validate an abi3 claim with:

```bash
python -m build --wheel --no-isolation
pipx run abi3audit --strict --report dist/*.whl
```

Also inspect every bundled `.so`, including helper extensions and vendored libraries. If any target uses `Python_add_library(... USE_SABI ...)`, pass `SKBUILD_SABI_VERSION` consistently and ensure headers are compiled with the limited API.

Do not confuse abi3 with the GNU C++ ABI. `_GLIBCXX_USE_CXX11_ABI`, libstdc++ symbol versions, CUDA major versions, and manylinux glibc floors remain separate compatibility constraints.

## Native ABI And RPATH

- Link `tvm_ffi::shared` and ensure its runtime library is discoverable from the installed package.
- Prefer target-local RPATH such as `$ORIGIN`-relative paths rather than absolute build paths.
- Inspect with `readelf -d`, `objdump -p`, `ldd -r`, and `nm -D --undefined-only`.
- Run wheel repair with explicit exclusions only for libraries guaranteed by the runtime environment.
- Test import from a clean virtual environment before importing Torch, then after importing Torch; results should not depend on accidental preloading unless documented.
