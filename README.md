# Kernel library template

Build a Linux CUDA kernel library for **Torch >= 2.13 and GIL-enabled CPython >= 3.12**. The default implementation uses nanobind's statically linked stable ABI mode and the stable LibTorch API. One wheel per architecture supports the tested Torch/Python versions without a separate binding or Torch compatibility backend package.

## Build and runtime versions

| Setting | Version | Purpose |
|---|---|---|
| Build dependency | `torch>=2.14` | Provides the headers used to compile the extension |
| `TORCH_TARGET_VERSION` | `0x020d000000000000` (2.13) | Limits direct Torch API references to the minimum runtime ABI |
| Runtime dependency | `torch>=2.13` | Permits installation on the minimum supported Torch |
| Python ABI | `cp312-abi3` | Uses the CPython stable ABI starting at 3.12 |

`pyproject.toml` declares the build and runtime requirements separately. CMake also checks the installed build-time Torch, including when build isolation is disabled. Building with newer headers does not permit directly linking newer Torch functions while retaining the 2.13 runtime floor.

The build uses `find_package(Torch 2.14 REQUIRED)` and links the `torch_cpu` library file without inheriting the full Torch target's dependencies. The CUDA kernel and launcher live together in `klib/ops/add/add_kernel.cu` and compile directly into `_stable.abi3.so`, with a static CUDA runtime. There is no separate kernel plugin or runtime kernel loader. ABI compatibility still depends on using only stable Torch and CPython APIs.

## Build and install

Install a CUDA toolkit with `nvcc` on `PATH` (or set `CUDACXX=/path/to/nvcc`), a C++20 compiler and CMake >= 3.27. For a build with isolation, pip installs the declared Python build dependencies:

```bash
python -m pip wheel . -w wheelhouse
```

To use an existing environment, install CUDA-enabled Torch >= 2.14, `nanobind==3.1.0` and `scikit-build-core`, then build:

```bash
python -m pip wheel . --no-build-isolation --no-deps -w wheelhouse
```

The default CUDA architectures are `8.0;8.6;8.9;9.0+PTX`. Override them through `TORCH_CUDA_ARCH_LIST`, which TorchConfig uses instead of `CMAKE_CUDA_ARCHITECTURES`:

```bash
python -m pip wheel . --no-build-isolation --no-deps -w wheelhouse \
  '-Ccmake.define.TORCH_CUDA_ARCH_LIST=9.0+PTX'
```

In a target environment with CUDA-enabled Torch already installed:

```bash
python -m pip install --no-deps /path/to/klib-0.1.0-cp312-abi3-linux_x86_64.whl
```

Nanobind is a build dependency only. No `nanobind-backend` or custom Torch backend wheel is needed.

## Tensor conversion

On Torch 2.14+, the extension resolves `torch_tensor_from_pyobject` and `torch_tensor_to_pyobject` dynamically from the installed Torch. They are not direct link dependencies, allowing the extension to load on 2.13.

On Torch 2.13, `compat/torch213/conversion.h` calls the installed Torch's private Python functions `torch._C._aoti.unsafe_alloc_void_ptr_from_tensor` and `alloc_tensor_by_stealing_from_void_ptr`. This does not invoke `torch.compile`. These private APIs need validation for supported Torch 2.13 distributions; they do not carry the official stable conversion API's compatibility guarantee.

The input function allocates a Tensor handle owned by `torch::stable::Tensor`. Despite its name, the singular output function in [Torch 2.13](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/inductor/aoti_runner/pybind.cpp#L234) copies the Tensor without freeing that handle. The wrapper therefore retains responsibility for deleting it. The plural output function has different ownership semantics and cannot be substituted directly.

## API

```python
import torch
import klib

x = torch.zeros(257, device="cuda")
klib.add(x)
assert torch.equal(x, torch.ones_like(x))
assert klib.sum(x).item() == 257
assert klib.roundtrip(x) is x

view = torch.empty_strided((10, 1), (100, 1), device=x.device)
y = klib.empty_like(view)
assert y.stride() == view.stride()
print(klib.interop_info())
```

- `add`: increments a contiguous float32 CUDA tensor in place. It rejects other devices and inputs requiring gradients, and increments the version counter so existing autograd graphs can detect mutation. The CUDA kernel checks bounds, including for empty and partial-block inputs.
- `empty_like`: preserves exact shape, strides, dtype and device, including layouts with gaps. It does not copy values or `requires_grad`.
- `sum`: uses `torch::stable::sum` and its native autograd behavior.
- `roundtrip`: converts a Tensor to a stable handle and back, preserving identity without copying device data.

The template targets CUDA kernels. Other kernel backends, custom autograd nodes, `record_stream`, custom Tensor subclasses, multi-GPU and `torch.compile` are outside the tested scope. Non-default-stream tests retain tensors until synchronization.

## Validate a wheel

Build once, then run the same wheel under each target Python/Torch environment:

```bash
python tests/check_wheel.py wheelhouse/klib-0.1.0-cp312-abi3-linux_x86_64.whl
abi3audit --strict --report wheelhouse/klib-0.1.0-cp312-abi3-linux_x86_64.whl
```

The tests require a GPU, CUDA-enabled Torch and PyTorch's internal test dependencies (including `expecttest`). The wheel runner extracts into a temporary directory, checks the package metadata, single-extension layout and loaded extension hash, and runs tests while prohibiting backend imports.

Tests cover conversion ownership and identity, invalid inputs, exact-stride allocation, sum gradients, mutation detection, kernel bounds and non-default streams. Removing the private conversion functions must break the 2.13 path while leaving the 2.14+ path operational.

Stable ABI compatibility does not eliminate OS/architecture, glibc/libstdc++, CUDA driver/toolkit or backend requirements. Wheels built on a development host use `linux_*` tags; manylinux release packaging and additional Torch/Python combinations require separate validation.

## TVM-FFI build option

The original implementation remains selectable with `-Ccmake.define.KLIB_IMPL=tvm_ffi`. Install `apache-tvm-ffi` and a CUDA toolchain, then build with `--no-build-isolation`. This mode retains `add` and `empty_like` via `klib/_tvm_impl.py`. It has not been rebuilt in this validation, and currently shares the stable mode's package metadata.
