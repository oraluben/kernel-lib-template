---
name: migrate-kernel-library-to-tvm-ffi
description: Audit and migrate CUDA/C++ operator or communication libraries from torch.utils.cpp_extension, pybind11, and libtorch bindings to tvm-ffi, normally together with CMake and scikit-build-core. Use when assessing tvm-ffi feasibility, removing compile-time Torch coupling, preserving a Python operator API, converting tensor allocation/view/stream handling, designing Python callbacks for unsupported ATen operations, packaging abi3 wheels, deciding whether CUDA stubs are justified, or validating a migration without changing kernel behavior.
---

# Migrate Kernel Libraries to TVM FFI

Treat the migration as a compatibility project, not a binding rewrite. Preserve the current branch's kernels, launch signatures, Python API, build-time generated files, wheel contents, and runtime behavior unless the user explicitly changes scope.

## Start With An Audit

1. Read repository instructions and check the worktree before editing.
2. Identify the exact baseline commit or branch. Do not copy whole host files from an older FFI branch; use old work only as a pattern reference.
3. Run the audit script and retain its JSON output for comparison:

```bash
python <skill-dir>/scripts/audit_operator_library.py /path/to/repo \
  --json-out /tmp/operator-before.json
```

4. Read [feasibility-and-impact.md](references/feasibility-and-impact.md). Classify every coupling point before choosing a design.
5. Produce a file-by-file plan that separates:
   - mechanical binding and tensor metadata conversion;
   - allocation and view conversion supported by tvm-ffi;
   - ATen behavior requiring a Python callback or an explicitly approved native implementation;
   - stream, ownership, and asynchronous lifetime work;
   - build, packaging, stubs, generated files, and Stable ABI work.

For a pure binding migration, make "do not add, delete, or modify CUDA kernels" the default constraint. Require explicit user approval to change it. List every planned Python callback by global name, arguments, return value, registration file, registration timing, and C++ call site before implementation.

Do not call the migration low risk merely because kernels are already plain CUDA. The largest failures usually occur in the host launch layer, generated/JIT source references, Python wrappers, and packaging.

## Choose The Migration Shape

Prefer two independently buildable stages when the repository is non-trivial:

1. Move `setup.py` build logic to CMake + scikit-build-core while retaining the existing binding.
2. Replace pybind11/libtorch with tvm-ffi on top of the verified CMake baseline.

Combine the stages only for small libraries with a thin, stateless binding and no special post-build actions.

Use these default choices:

- Accept borrowed inputs as `tvm::ffi::TensorView` only when their lifetime need not outlive the call.
- Accept or retain `tvm::ffi::Tensor` when asynchronous work or deferred release must keep the owner alive.
- Return owning tensors allocated with `Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, ...)`.
- Create zero-copy views with `as_strided`; make the returned view retain its backing allocation.
- Obtain the framework's current CUDA stream with `TVMFFIEnvGetStream`.
- Add no `record_stream` mechanism for work confined to the current stream.
- For cross-stream work, implement explicit ownership retention until CUDA events complete, or use a current-version Torch bridge only when that dependency is intentional. Read [tvm-ffi-migration-patterns.md](references/tvm-ffi-migration-patterns.md) before choosing.
- Keep simple allocation/view operations native to tvm-ffi. Register complex Torch operations in Python and call them through `Function::GetGlobalRequired` when exact ATen semantics must be preserved without linking libtorch.
- Represent stateful C++ resources with a real tvm-ffi object when practical; otherwise use a synchronized opaque-handle registry with explicit destruction and Python ownership wrappers.

## Preserve The Contract

Before editing, freeze the following surfaces:

- exported native function names and positional arity;
- Python signatures, defaults, keyword arguments, overload behavior, and return structure;
- accepted dtypes, devices, shapes, strides, and non-contiguous cases;
- CUDA architecture list, compile flags, source list, separable compilation, and device-link behavior;
- generated files, JIT headers, SASS patching, spill checks, and cached-wheel behavior;
- wheel package contents, RPATH, external shared-library policy, and type stubs.

Export raw tvm-ffi functions privately and keep explicit Python wrappers for the public API. Do not use a generic `inspect`-based kwargs adapter on hot paths.

## Migrate Mechanically

Read [tvm-ffi-migration-patterns.md](references/tvm-ffi-migration-patterns.md), then convert one tightly coupled unit at a time.

1. Add a self-contained compatibility header for dtype, device, shape, stride, allocation, view, CUDA error, and device-guard helpers.
2. Convert public entry signatures and registration without changing kernel or launcher signatures.
3. Convert validation before pointer extraction so invalid inputs fail clearly.
4. Convert allocations and views while preserving dtype, device, shape, stride, byte offset, zero initialization, and ownership.
5. Replace current-stream and device queries with tvm-ffi/CUDA equivalents.
6. Add Python callbacks only for operations whose semantics are not already covered by tvm-ffi or simple CUDA runtime calls. Register callbacks before loading or invoking native functions.
7. Keep original kernel source files and host launch parameter order. Verify generated/JIT includes and launch arity separately; ordinary C++ compilation cannot detect all such mismatches.
8. Remove Torch, ATen, c10, and pybind11 includes and links only after all call sites have been converted.
9. Do not hide import, callback registration, or binding failures with broad `except Exception: pass` blocks.
10. Remove migration-created dead code and verify that every new host helper, runtime class, and source file has a caller or build/include path.

Compile each converted unit or tightly coupled group before moving on. Preserve evidence for every claim that a problem is fixed. Commit checkpoints only when the user requests commits or repository workflow requires them.

## Build And Package

Read [build-packaging.md](references/build-packaging.md) before writing CMake or `pyproject.toml`.

- Translate every behavior from `setup.py`; do not translate only the source list and compiler flags.
- Prefer explicit source manifests. Use scoped `CONFIGURE_DEPENDS` globs only for generated instantiation directories with a stable inclusion rule.
- Discover `tvm_ffi` through the build interpreter, then `find_package(tvm_ffi CONFIG REQUIRED)`.
- Build a normal shared library loaded by `tvm_ffi.load_module` unless CPython symbols are genuinely required.
- Set `wheel.py-api = "cp38"` only after confirming the wheel has no dependency on newer CPython APIs. Treat the abi3 tag as a claim requiring `abi3audit`, not as a build optimization.
- Keep direct CUDA shared-library links by default. Add stubs only when they solve a documented manylinux/link-time problem and verify every unresolved symbol and runtime lookup path.

## Validate In Layers

Read and follow [validation.md](references/validation.md).

At minimum:

1. Re-run the audit and compare it with the baseline:

```bash
python <skill-dir>/scripts/audit_operator_library.py /path/to/repo \
  --json-out /tmp/operator-after.json
python <skill-dir>/scripts/compare_operator_audits.py \
  /tmp/operator-before.json /tmp/operator-after.json
```

2. Configure and compile with the intended build interpreter and CUDA toolchain.
3. Build a wheel, inspect its file list and tags, run `ldd -r`/`readelf`/`nm`, and run `abi3audit` when claiming abi3.
4. Import from an installed wheel, not only from the source tree.
5. Compare exported names and arity, Python signatures, `.pyi`, generated/JIT references, and native dependencies.
6. Run correctness, edge-case, multi-stream, architecture, and performance tests on suitable GPUs.

State validation precisely. A CPU-only configure check, host syntax check, or successful wheel build is not GPU correctness validation.

## Iterate The Skill

When applying this skill exposes a missing category or false assumption, update the reusable workflow or references, then rerun the audit and validation. Record repository-specific outcomes in [case-studies.md](references/case-studies.md) only when they teach a general lesson.
