# Migration Validation

## Contents

- Baseline capture
- Static validation
- Build validation
- Wheel and ABI validation
- Runtime and GPU validation
- Acceptance report

## Baseline Capture

Record before editing:

- commit, submodule revisions, and worktree status;
- native exports and public Python signatures;
- source manifest, architectures, and compiler/link flags;
- generated files and wheel contents;
- representative correctness outputs and performance numbers;
- dynamic dependencies and wheel tag;
- JIT-generated include paths, symbols, and launch argument counts.

Keep the audit JSON produced by `audit_operator_library.py`.

## Static Validation

Run targeted searches outside vendored code:

```bash
rg -n 'torch::|at::|c10::|pybind11|torch/extension|torch/python' csrc
rg -n 'TVM_FFI_DLL_EXPORT_TYPED_FUNC' csrc
rg -n 'load_module|register_global_func' package_name
```

Require:

- no unintended Torch/ATen/c10/pybind11 compile-time dependencies;
- self-contained compatibility headers;
- exact export-name preservation;
- exact native positional arity preservation;
- no changed device kernel files unless explicitly authorized;
- no dead host files or unreferenced runtime classes introduced by the migration.

Compare before/after audits and investigate every removed export or newly detected high-risk pattern.

For runtime-generated source, use the bundled checker when applicable:

```bash
python <skill-dir>/scripts/check_generated_includes.py \
  /path/to/repo/csrc/jit_kernels/impls \
  --include-root /path/to/repo/package/include \
  --expand 'sm{}=90,100'
```

## Build Validation

Validate progressively:

1. CMake configure with the intended Python and CUDA toolchain.
2. Host translation-unit syntax or object compilation where possible.
3. Full CUDA compile and device link for every architecture group.
4. Required custom post-build actions.
5. Editable install and wheel build.

Capture the exact command, interpreter, compiler, CUDA version, and result. Do not report a configure-only result as a successful build.

For large libraries, keep a fast developer target that compiles the binding and a representative kernel, but still run the complete source manifest before acceptance.

## Wheel And ABI Validation

Inspect the wheel:

```bash
python -m zipfile -l dist/*.whl
unzip -p dist/*.whl '*/WHEEL'
auditwheel show dist/*.whl
pipx run abi3audit --strict --report dist/*.whl
```

Inspect every native library:

```bash
readelf -d path/to/lib.so
ldd -r path/to/lib.so
nm -D --undefined-only path/to/lib.so
```

Verify:

- expected Python files, JIT headers, `.pyi`, and native libraries are present once;
- no absolute build-tree RPATH remains;
- no unintended Torch libraries are needed;
- tvm-ffi and CUDA dependencies resolve according to policy;
- wheel tag matches the actual CPython API usage and manylinux floor.

Treat `abi3audit` and `auditwheel` as independent checks. A wheel can be valid `cp38-abi3` while still being restricted to a recent glibc/libstdc++ baseline or carrying the generic `linux_x86_64` platform tag. Report both results instead of describing abi3 as general wheel portability.

On a host without an NVIDIA driver, use the official CUDA driver stub only for loader/static-registration checks. Ensure its SONAME is discoverable as `libcuda.so.1`, then state explicitly that no kernel was launched.

Install into a clean environment and import from outside the source tree. Test both before and after importing Torch to catch accidental dependency preloading.

## Runtime And GPU Validation

Test on every supported architecture family.

### Contract tests

- positional and keyword calls;
- default and optional parameters;
- return tuple/list shape and `None` normalization;
- dtype, device, shape, stride, contiguity, and error messages;
- preallocated output and internally allocated output paths;
- zero-size and boundary-size inputs where supported.

### Correctness tests

- compare with the same reference used before migration;
- cover every dispatch branch and optional feature;
- test non-contiguous tensors when the old API supported them;
- test repeated import and repeated calls for ownership leaks.

### Stream and lifetime tests

- current non-default stream;
- temporary inputs released immediately after the call;
- secondary-stream use and early Python reference release for cross-stream libraries;
- allocator reuse pressure;
- CUDA Graph capture when supported;
- explicit destroy and interpreter shutdown for stateful resources.

### Runtime-generated code

- enumerate generated include paths and confirm each exists;
- compare kernel template and runtime launch arity with the current kernel signature;
- exercise cold-cache and warm-cache paths;
- verify cache keys include inputs that affect generated code or ABI.

### Performance

- compare launch overhead and representative latency/throughput;
- check register spills and local memory;
- ensure Python callbacks are not inserted into hot inner loops;
- explain any synchronization or allocation behavior change.

## Acceptance Report

Report validation in a table:

| Layer | Environment | Command/test | Result | Remaining gap |
| --- | --- | --- | --- | --- |

Use precise completion language:

- "static audit passed" means searches and export comparison passed;
- "wheel built" means packaging completed, not that kernels ran;
- "GPU correctness passed on SM90" does not cover SM100;
- "abi3audit passed" covers CPython Stable ABI, not CUDA or libstdc++ ABI;
- mark missing hardware or toolchains as unvalidated, not implicitly successful.
