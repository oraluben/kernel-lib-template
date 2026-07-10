# TVM FFI Migration Patterns

## Contents

- Boundary types
- Tensor metadata and validation
- Allocation and views
- Stream and lifetime semantics
- Python callbacks
- Stateful resources
- Registration and Python wrappers

## Boundary Types

Use the narrowest type that preserves ownership.

| Existing type | Preferred replacement |
| --- | --- |
| `const at::Tensor&` read during call | `const tvm::ffi::TensorView&` |
| Tensor used by asynchronous work after return | `const tvm::ffi::Tensor&` plus deferred ownership retention |
| `std::optional<at::Tensor>` | `std::optional<tvm::ffi::TensorView>` or owning `Tensor` |
| Native output tensor | `tvm::ffi::Tensor` |
| `std::vector<at::Tensor>` return | `tvm::ffi::Array<tvm::ffi::Tensor>` or a supported tuple |
| scalar dtype enum | `DLDataType` |
| CUDA device | `DLDevice` |

Include every tvm-ffi header a compatibility header needs. Do not rely on include order.

## Tensor Metadata And Validation

Map metadata mechanically:

```cpp
int ndim = tensor.ndim();
int64_t size0 = tensor.shape().at(0);
int64_t stride0 = tensor.stride(0);
void* ptr = tensor.data_ptr();
DLDataType dtype = tensor.dtype();
DLDevice device = tensor.device();
```

Normalize negative dimensions before indexing. A contiguous check must account for size-one dimensions and zero-element tensors according to the original contract.

Define named dtype constants and compare all DLPack fields:

```cpp
constexpr DLDataType kBF16{kDLBfloat, 16, 1};

inline bool same_dtype(DLDataType a, DLDataType b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}
```

Use `TVM_FFI_CHECK(..., ValueError)` or a project error macro for user input. Reserve internal checks for invariants.

## Allocation And Views

Allocate through the active framework environment:

```cpp
auto out = tvm::ffi::Tensor::FromEnvAlloc(
    TVMFFIEnvTensorAlloc,
    tvm::ffi::ShapeView(shape.data(), shape.size()),
    dtype,
    device);
```

Important properties:

- Allocation is uninitialized. Replace `torch::zeros` with allocation plus `cudaMemsetAsync` on the actual launch stream.
- Use the input tensor's exact dtype and device when replacing `options()`.
- Compute backing storage size carefully for custom strides and byte offsets.
- Use `as_strided` for views so the returned tensor keeps the backing object alive.
- Implement slice as pointer/byte-offset plus `as_strided`; do not use a global or `thread_local` tensor merely to extend lifetime.

Prefer small helper functions such as `empty_like`, `empty`, `zeros`, and `slice` over a broad Torch compatibility class for thin libraries. A compatibility tensor class is justified when hundreds of host call sites share the same small Torch API subset, but its ownership and conversion behavior must be explicit.

## Stream And Lifetime Semantics

Get the framework stream for the tensor's device:

```cpp
auto stream = static_cast<cudaStream_t>(
    TVMFFIEnvGetStream(tensor.device().device_type, tensor.device().device_id));
```

Use a CUDA device guard around device-specific runtime calls and launches when inputs may target a non-current device.

### Current stream only

If allocations, kernel launches, and consumers use the same current stream, no extra `record_stream` mechanism is needed. Preserve asynchronous execution.

### Cross-stream use

`FromEnvAlloc` may use the caller framework's caching allocator. It does not by itself express that a tensor was later used on a different stream.

Prefer framework-independent deferred ownership retention:

1. Hold an owning `tvm::ffi::Tensor` or DLPack owner reference.
2. Record an event after all work enqueued on each secondary stream.
3. Retain the owner until every event reports completion.
4. Recycle events and poll at API entry/exit.
5. Define CUDA Graph capture behavior; event query and allocation behavior may differ during capture.

This is semantically equivalent to preventing the underlying allocation from being released too early. It avoids linking libtorch and survives Torch internal ABI changes.

Use a runtime-compiled current-Torch bridge only when direct caching-allocator integration is required. Compile it with the active Torch headers and flags, load it explicitly, cache by Torch/CUDA/compiler ABI, and fail clearly. Do not assume `recordStream` is available through `dlsym`; common implementations are inline wrappers over allocator virtual methods.

## Python Callbacks

Keep simple allocation and view work in C++. For operations such as implicit dtype-converting `copy_`, `index_select`, complex bitwise/view chains, or nuanced non-contiguous ATen behavior, register exact Torch implementations in Python:

```python
@tvm_ffi.register_global_func("my_lib.copy_convert", override=True)
def _copy_convert(dst, src):
    dst.copy_(src)
    return dst
```

Call them from C++:

```cpp
static auto callback =
    tvm::ffi::Function::GetGlobalRequired("my_lib.copy_convert");
callback(dst, src);
```

Requirements:

- Register callbacks before loading or invoking dependent native functions.
- Namespace names by package.
- Let missing callbacks fail at import or first use with a specific error.
- Mark callback sites consistently for later removal.
- Keep Torch as a Python runtime dependency while callbacks remain, even after removing libtorch from the native build.
- Do not replace ATen behavior with a new CUDA kernel during a pure FFI migration. That changes the numerical and layout implementation surface and requires separate scope and validation.

## Stateful Resources

For classes owning GPU or communication resources, choose one of:

- tvm-ffi object/reflection types with finalization;
- opaque integer handles stored in a synchronized registry;
- thin Python owners that call explicit native destroy functions and use `weakref.finalize` as a fallback.

Never expose a raw pointer as an unchecked integer without generation or registry validation. Define behavior for double destroy, interpreter shutdown, fork, concurrent calls, and exceptions during construction.

## Registration And Python Wrappers

Register raw functions:

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_select, topk_select_impl);
```

Load the library from Python and preserve the public API in Python:

```python
_LIB = tvm_ffi.load_module(path)

def topk_select(input, topk, *, sorted=False):
    return _LIB.topk_select(input, topk, sorted)
```

Keep keyword-only behavior, defaults, overloads, output normalization, and dataclass mutation in Python. Generate `.pyi` from this stable public layer or maintain a checked static stub. Compare stub names and arity with the public API during validation.

Avoid broad exception swallowing around callback registration or module loading. Missing exports and callbacks should fail at a deterministic boundary with the original exception context.
