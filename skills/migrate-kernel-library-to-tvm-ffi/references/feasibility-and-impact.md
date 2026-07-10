# Feasibility And Impact

## Contents

- Feasibility model
- Coupling classification
- Impact surface
- Planning output

## Feasibility Model

Assess feasibility by host behavior, not by whether the device kernels use Torch.

| Library shape | Typical result | Main work |
| --- | --- | --- |
| Stateless op, caller allocates outputs, current stream only | High feasibility, low risk | Tensor metadata, checks, registration, build |
| Stateless op, native outputs and simple views | High feasibility, medium risk | Env allocation, ownership, dtype/device fidelity |
| ATen transforms around CUDA launches | Medium feasibility | Python callbacks or approved native equivalents |
| Stateful objects with CUDA/NCCL/NVSHMEM resources | Medium feasibility, high design risk | Object lifetime, handles, thread safety, destruction |
| Cross-stream asynchronous communication | Medium feasibility, high correctness risk | Events, ownership retention, allocator semantics |
| Runtime-generated CUDA source or launch metadata | Feasible but high regression risk | Include/symbol/launch-arity validation |

Do not treat NCCL, NVSHMEM, or CUDA as Torch-bound merely because the current binding uses `torch::Tensor`. Their C APIs are independent. Identify the convenience services currently supplied by Torch: allocation, dtype/shape inspection, stream selection, device guards, event wrappers, error translation, and Python object conversion.

## Coupling Classification

Classify every call site into one primary category.

1. **Binding only**: `PYBIND11_MODULE`, `m.def`, `pybind11::class_`, STL conversion.
2. **Tensor metadata**: size, stride, dtype, device, contiguity, pointer extraction.
3. **Allocation/view**: empty, zeros, empty_like, from_blob, view, reshape, slice, as_strided.
4. **ATen computation**: copy with conversion, index_select, reductions, bitwise expressions, broadcasting.
5. **Stream/device**: current stream, device guard, stream pool, events, record_stream.
6. **State/lifetime**: classes owning communicators, buffers, streams, events, or JIT modules.
7. **Build/package**: Torch discovery, C++ ABI flags, extension suffix, RPATH, generated files, wheel repair.
8. **Runtime generation**: JIT source includes, template parameters, driver launches, SASS patching.

Recommended dispositions:

- Categories 1-3 are normally direct tvm-ffi conversions.
- Category 4 requires semantic review. Use Python callbacks when exact Torch behavior matters and native replacement is not explicitly in scope.
- Category 5 is direct only for current-stream work. Cross-stream use requires a lifetime design.
- Categories 6 and 8 drive the risk rating even if occurrence counts are small.

## Impact Surface

List impact by contract, not only by files changed.

### Native ABI and dependencies

- Remove `libtorch`, `libtorch_python`, `libc10`, and Torch headers from compile/link inputs.
- Add the tvm-ffi C++ headers and shared runtime.
- Preserve CUDA, NCCL, NVSHMEM, CUTLASS, and project-specific native dependencies.
- Recheck `_GLIBCXX_USE_CXX11_ABI`; do not hard-code a value inherited from Torch after Torch is removed.

### Python API

- Native tvm-ffi packed functions are position-oriented; preserve keyword arguments in explicit Python wrappers.
- Preserve `None`, tuple/list shape, empty optional outputs, dataclass mutations, and overload behavior.
- Load `.so` files with `tvm_ffi.load_module`; do not assume `import package._C` remains valid.
- Preserve type stubs or regenerate them from the public Python layer.

### Tensor semantics

- Preserve shape, logical strides, byte offsets, dtype lanes, device id, and contiguity acceptance.
- Distinguish borrowed views from owning tensors.
- Preserve zero initialization and implicit dtype conversions; `FromEnvAlloc` is uninitialized.
- Preserve non-contiguous behavior or reject it explicitly before launch.

### Runtime behavior

- Preserve current device and current stream selection.
- Preserve asynchronous behavior; do not insert synchronization merely to avoid lifetime analysis.
- Preserve JIT cache keys, generated headers, launch parameters, post-link patches, and spill checks.

### Distribution

- Preserve files copied by custom build commands, including headers used by runtime compilation.
- Reevaluate manylinux repair exclusions and external CUDA libraries.
- Verify the wheel tag and CPython dependency claim separately from C++ library compatibility.

## Planning Output

Before editing, produce a compact table with these columns:

| File or surface | Current dependency | Category | Replacement | Risk | Validation |
| --- | --- | --- | --- | --- | --- |

Also list:

- raw native exports and their arity;
- public Python signatures and keyword arguments;
- allocations and zero-initialized allocations;
- view/slice operations;
- cross-stream tensor uses;
- Python callbacks and registration order;
- generated files and post-build actions;
- GPU architectures and required validation hardware.

For each callback, record:

| Global name | Python signature | Return | Registration timing | C++ caller | Removal plan |
| --- | --- | --- | --- | --- | --- |
