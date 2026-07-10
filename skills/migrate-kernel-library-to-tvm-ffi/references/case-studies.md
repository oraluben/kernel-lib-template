# Case Studies

## Contents

- Kernel library template
- TopK Select
- Exact TopK Attention
- General lessons

## Kernel Library Template

The template demonstrates the low-risk target architecture:

- scikit-build-core drives CMake;
- CMake builds ordinary shared libraries loaded by `tvm_ffi.load_module`;
- `TensorView` carries input metadata;
- `Tensor::FromEnvAlloc` creates framework-owned output tensors;
- `as_strided` creates an owning view;
- `wheel.py-api = "cp38"` and `abi3audit` express and verify the Stable ABI claim;
- pip-installed CUDA toolchain discovery is separated from operator code.

The template does not demonstrate cross-stream allocator lifetime, stateful resources, or setup.py post-build migration.

## TopK Select

Initial audit classifies this repository as a thin stateless operator:

- two native exports;
- caller-preallocated outputs;
- no C++ tensor allocation;
- one current-stream CUDA launch path;
- Torch coupling concentrated in one host API file, dispatch macros, and `setup.py`;
- a custom register-spill check that must survive the build migration;
- 182 generated kernel instantiation translation units.

The migration confirmed the low-to-medium assessment:

- pybind11 and Torch C++ compile/link dependencies were removed without changing kernels;
- CMake + scikit-build-core compiled all 182 generated CUDA translation units;
- the two export names and arity were preserved behind explicit Python wrappers;
- the spill check and `ldd -r` passed;
- the wheel was tagged `cp38-abi3`, and `abi3audit --strict` passed;
- the final rebuilt wheel was `topk_select-2.0.2+b7a6d5b.20260710.91356-cp38-abi3-linux_x86_64.whl`;
- `auditwheel show` reported a `manylinux_2_39_x86_64` native baseline despite the valid abi3 tag.

No suitable GPU was available, so SM100/SM103 numerical correctness and performance remain unvalidated.

## Exact TopK Attention

Initial audit classifies this repository as a larger allocating operator library:

- five native exports;
- 55 `at::Tensor` host references and 25 native allocations;
- slice/view and zero-initialization behavior;
- current-stream launches with device guards;
- 141 CUDA translation units;
- optional preallocated outputs;
- post-build SASS patching;
- mutable Python scheduler metadata;
- Hopper and Blackwell dispatch branches.

The migration confirmed the medium-to-high assessment. A small owning tensor compatibility layer replaced the repeated host-side Torch subset:

- `Tensor::FromEnvAlloc` handles native outputs and intermediate buffers;
- owning `as_strided` views preserve slice, transpose, and backing-storage lifetime;
- zero-filled buffers use `cudaMemsetAsync` on `TVMFFIEnvGetStream`;
- device guards and current-stream launches were preserved;
- mutable decode scheduler metadata remains a Python wrapper concern;
- no `csrc/kernels/` file changed.

Validation completed with CUDA 13.1:

- all 141 kernel translation units plus the host insight source compiled for SM90a, SM100a, and SM103a;
- link, 16-kernel SASS matching/patching, and the register-spill check passed;
- the five exports preserved names and arity `10, 13, 14, 15, 17`;
- public Python signature ASTs, including defaults and kwargs, matched the baseline;
- the installed wheel library passed stub-assisted `ldd -r` and loaded all five exports without Torch, c10, pybind11, or CPython symbols;
- `exact_topk_attn-3.8.1+b818557.20260710.85817-cp38-abi3-linux_x86_64.whl` passed `abi3audit --strict` with a Python 3.8 baseline;
- `auditwheel show` reported a `manylinux_2_39_x86_64` native baseline, demonstrating that abi3 does not imply broad glibc/libstdc++ compatibility.

The first wheel attempt OOMed because a 384-core builder combined unconstrained Ninja jobs with nvcc `--threads=32`. Retrying with `CMAKE_BUILD_PARALLEL_LEVEL=8` and `NVCC_THREADS=2` reused completed work and succeeded. No suitable GPU or real Torch runtime was available, so numerical correctness, allocator behavior under actual Torch, scheduler reuse during launches, and performance remain unvalidated.

Neither TopK Select nor Exact TopK Attention uses a private communication stream in the audited host API. `TVMFFIEnvGetStream` is sufficient and a DeepEP-style deferred-release pool is unnecessary unless later code introduces cross-stream work.

## General Lessons

- Occurrence count is not the main risk signal; a single JIT include or launch-arity mismatch can be worse than dozens of mechanical tensor conversions.
- Removing libtorch from the native library does not imply removing Torch from Python runtime dependencies.
- Build migration must preserve custom actions, not only compilation.
- Stable Python API, CPython Stable ABI, and native CUDA/C++ ABI are separate claims.
- Stubs are a deployment decision, not a default ingredient of tvm-ffi migration.
- An older FFI branch is safe as a syntax-pattern reference only. Copying whole host files can silently revert current kernel features, generated include paths, and launch argument lists.
- Export-name equality is insufficient; compare positional arity and Python keyword compatibility.
- During a pure FFI migration, use callbacks for unsupported ATen semantics instead of introducing replacement kernels.
- Bound both build-system parallelism and nvcc internal parallelism for large source manifests.
- A successful abi3 audit says nothing about the wheel's glibc, libstdc++, or CUDA compatibility floor.
- Driver stubs are useful for CPU-only loader checks, but cannot substitute for GPU execution.
