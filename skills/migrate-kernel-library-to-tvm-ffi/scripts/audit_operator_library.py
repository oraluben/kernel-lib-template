#!/usr/bin/env python3
"""Audit a CUDA/C++ operator library before or after a tvm-ffi migration."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_SUFFIXES = {".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp"}
PYTHON_SUFFIXES = {".py"}
BUILD_NAMES = {"setup.py", "pyproject.toml", "CMakeLists.txt"}
EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "build",
    "dist",
    "_skbuild",
    "__pycache__",
    "3rdparty",
    "third_party",
    "third-party",
    "cutlass",
}

CATEGORY_PATTERNS = {
    "binding": re.compile(
        r"pybind11|PYBIND11_MODULE|TORCH_EXTENSION_NAME|m\s*\.\s*def\s*\(|"
        r"torch/(?:extension|python)\.h"
    ),
    "tensor_types": re.compile(r"\b(?:torch|at)::Tensor\b|\bc10::optional\s*<\s*(?:torch|at)::Tensor"),
    "allocation": re.compile(
        r"\btorch::(?:empty|zeros|ones|full|empty_like|zeros_like|from_blob)\s*\(|"
        r"\bTensor::FromEnvAlloc\s*\(|\bTVMFFIEnvTensorAlloc\b"
    ),
    "tensor_ops": re.compile(
        r"\.(?:copy_|index_select|view|reshape|slice|unsqueeze|squeeze|transpose|permute)\s*\(|"
        r"\btorch::(?:index_select|bitwise_|sum|cat|stack)"
    ),
    "stream_device": re.compile(
        r"getCurrentCUDAStream|getStreamFromPool|CUDAStream|CUDAGuard|getCurrentDeviceProperties|"
        r"getDeviceProperties|TVMFFIEnvGetStream|TVMFFIEnvSetStream"
    ),
    "cross_stream_lifetime": re.compile(r"record_stream|recordStream|ReleasePool|cudaStreamWaitEvent"),
    "dtype": re.compile(r"(?:torch|at)::k[A-Z]|ScalarType|DLDataType|kDLBfloat|kDLFloat|kDLInt|kDLUInt"),
    "ffi": re.compile(
        r"TVM_FFI_DLL_EXPORT_TYPED_FUNC|tvm::ffi::(?:Tensor|TensorView|Function)|"
        r"tvm_ffi\.(?:load_module|register_global_func)"
    ),
    "runtime_generation": re.compile(
        r"cudaLaunchKernel|cuLaunchKernel|nvcc|cuobjdump|patch_sass|jit[_:]|JIT|generated code|"
        r"std::format\s*\([^\n]*(?:#include|\.cuh)"
    ),
    "stateful": re.compile(
        r"pybind11::class_|std::unique_ptr<|std::shared_ptr<|cudaEventCreate|cudaStreamCreate|"
        r"ncclComm|nvshmem|opaque.?handle|handle_registry"
    ),
}

PYBIND_EXPORT_RE = re.compile(
    r"\.def\s*\(\s*\"([A-Za-z_][A-Za-z0-9_.]*)\"\s*,\s*&?([A-Za-z_][A-Za-z0-9_:]*)",
    re.S,
)
FFI_EXPORT_RE = re.compile(
    r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
    r"([A-Za-z_][A-Za-z0-9_:]*)\s*\)",
    re.S,
)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        if path.suffix in SOURCE_SUFFIXES | PYTHON_SUFFIXES or path.name in BUILD_NAMES:
            yield path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def mask_cpp_comments(text: str) -> str:
    """Replace C/C++ comments with spaces while preserving line numbers."""
    result = list(text)
    in_string: str | None = None
    in_line_comment = False
    in_block_comment = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            else:
                result[index] = " "
            index += 1
            continue
        if in_block_comment:
            if char == "*" and next_char == "/":
                result[index] = " "
                result[index + 1] = " "
                in_block_comment = False
                index += 2
            else:
                if char != "\n":
                    result[index] = " "
                index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            index += 1
            continue
        if char in {'"', "'"}:
            in_string = char
            index += 1
            continue
        if char == "/" and next_char == "/":
            result[index] = " "
            result[index + 1] = " "
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            result[index] = " "
            result[index + 1] = " "
            in_block_comment = True
            index += 2
            continue
        index += 1
    return "".join(result)


def line_matches(text: str, pattern: re.Pattern[str]) -> list[int]:
    return [index for index, line in enumerate(text.splitlines(), 1) if pattern.search(line)]


def count_parameters(parameters: str) -> int | None:
    value = parameters.strip()
    if not value or value == "void":
        return 0
    depths = {"(": 0, "[": 0, "{": 0, "<": 0}
    matching = {")": "(", "]": "[", "}": "{", ">": "<"}
    commas = 0
    in_string: str | None = None
    in_line_comment = False
    in_block_comment = False
    escaped = False
    index = 0
    while index < len(value):
        char = value[index]
        next_char = value[index + 1] if index + 1 < len(value) else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            index += 1
            continue
        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                index += 2
            else:
                index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            index += 1
            continue
        if char == "/" and next_char == "/":
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            in_block_comment = True
            index += 2
            continue
        if char in {'"', "'"}:
            in_string = char
        elif char in depths:
            depths[char] += 1
        elif char in matching:
            opener = matching[char]
            depths[opener] = max(0, depths[opener] - 1)
        elif char == "," and not any(depths.values()):
            commas += 1
        index += 1
    return commas + 1


def function_arity(text: str, target: str) -> int | None:
    leaf = target.rsplit("::", 1)[-1]
    pattern = re.compile(rf"(?:^|\n)[^\n;{{}}]*\b{re.escape(leaf)}\s*\(", re.M)
    for match in pattern.finditer(text):
        open_pos = text.find("(", match.start())
        depth = 0
        in_string: str | None = None
        escaped = False
        close_pos = None
        for index in range(open_pos, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
                continue
            if char in {'"', "'"}:
                in_string = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close_pos = index
                    break
        if close_pos is None:
            continue
        suffix = text[close_pos + 1 : close_pos + 160]
        if not re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?(?:->[^\{]+)?\{", suffix):
            continue
        return count_parameters(text[open_pos + 1 : close_pos])
    return None


class PythonVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.native_aliases: set[str] = set()
        self.keyword_calls: list[dict[str, Any]] = []
        self.public_functions: list[dict[str, Any]] = []
        self.load_modules: list[int] = []
        self.global_registrations: list[int] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            if alias.name.endswith("_cuda") or local in {"_C", "_LIB"}:
                self.native_aliases.add(local)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            local = alias.asname or alias.name
            if local in {"_C", "_LIB"} or local.endswith("_cuda"):
                self.native_aliases.add(local)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not node.name.startswith("_"):
            positional = len(node.args.posonlyargs) + len(node.args.args)
            defaults = len(node.args.defaults)
            self.public_functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "positional": positional,
                    "required_positional": positional - defaults,
                    "keyword_only": [arg.arg for arg in node.args.kwonlyargs],
                }
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = dotted_name(node.func)
        if dotted.endswith("tvm_ffi.load_module") or dotted == "tvm_ffi.load_module":
            self.load_modules.append(node.lineno)
        if "register_global_func" in dotted:
            self.global_registrations.append(node.lineno)
        if node.keywords:
            root = dotted.split(".")[0]
            if root in self.native_aliases or root in {"_C", "_LIB"} or root.endswith("_cuda"):
                self.keyword_calls.append(
                    {
                        "line": node.lineno,
                        "callee": dotted,
                        "keywords": [kw.arg for kw in node.keywords if kw.arg is not None],
                    }
                )
        self.generic_visit(node)


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def analyze_python(path: Path, text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as err:
        return {"syntax_error": f"{err.msg} at line {err.lineno}"}
    visitor = PythonVisitor()
    visitor.visit(tree)
    return {
        "native_aliases": sorted(visitor.native_aliases),
        "keyword_calls": visitor.keyword_calls,
        "public_functions": visitor.public_functions,
        "load_module_lines": visitor.load_modules,
        "global_registration_lines": visitor.global_registrations,
    }


def detect_build(root: Path, texts: dict[Path, str]) -> dict[str, Any]:
    setup = texts.get(root / "setup.py", "")
    pyproject = texts.get(root / "pyproject.toml", "")
    cmake = texts.get(root / "CMakeLists.txt", "")
    return {
        "has_setup_py": bool(setup),
        "has_pyproject": bool(pyproject),
        "has_cmake": bool(cmake),
        "torch_cpp_extension": bool(re.search(r"BuildExtension|CUDAExtension|CppExtension", setup)),
        "scikit_build_core": "scikit_build_core.build" in pyproject,
        "tvm_ffi_build_dependency": "apache-tvm-ffi" in pyproject,
        "wheel_py_api": next(
            iter(re.findall(r"wheel\.py-api\s*=\s*[\"']([^\"']+)", pyproject)), None
        ),
        "find_tvm_ffi": bool(re.search(r"find_package\s*\(\s*tvm_ffi", cmake, re.I)),
        "find_torch": bool(re.search(r"find_package\s*\(\s*Torch", cmake)),
        "python_add_library": "Python_add_library" in cmake,
        "custom_actions": sorted(
            name
            for name, pattern in {
                "sass_patch": r"patch_sass|patch_f2fp|mmap\.mmap",
                "spill_check": r"spill|res-usage",
                "generated_stubs": r"\.pyi|generate_pyi",
                "generated_python": r"configure_file|envs\.py|generate_default_envs",
                "cached_wheels": r"CachedWheels|cached.?wheel",
            }.items()
            if re.search(pattern, setup + "\n" + cmake, re.I)
        ),
    }


def risk_assessment(categories: Counter[str], build: dict[str, Any], exports: dict[str, list[str]]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    weighted = {
        "binding": 1,
        "tensor_types": 1,
        "allocation": 2,
        "tensor_ops": 3,
        "stream_device": 2,
        "cross_stream_lifetime": 5,
        "runtime_generation": 4,
        "stateful": 5,
    }
    for category, weight in weighted.items():
        count = categories.get(category, 0)
        if count:
            score += min(count, 5) * weight
            if category in {"allocation", "tensor_ops", "cross_stream_lifetime", "runtime_generation", "stateful"}:
                reasons.append(f"{category}: {count} matching lines")
    if build["custom_actions"]:
        score += 2 * len(build["custom_actions"])
        reasons.append("custom build actions: " + ", ".join(build["custom_actions"]))
    if len(exports["pybind"]) > 10:
        score += 5
        reasons.append(f"wide native export surface: {len(exports['pybind'])}")
    level = "low" if score < 15 else "medium" if score < 40 else "high"
    return {"score": score, "level": level, "reasons": reasons}


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = sorted(iter_files(root))
    texts = {path: read_text(path) for path in files}
    category_counts: Counter[str] = Counter()
    findings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exports: dict[str, Any] = {
        "pybind": [],
        "tvm_ffi": [],
        "pybind_details": {},
        "tvm_ffi_details": {},
    }
    python: dict[str, Any] = {}
    source_counts: Counter[str] = Counter()

    for path, text in texts.items():
        rel = str(path.relative_to(root))
        if path.suffix in SOURCE_SUFFIXES:
            source_without_comments = mask_cpp_comments(text)
            source_counts[path.suffix] += 1
            for name, target in PYBIND_EXPORT_RE.findall(source_without_comments):
                exports["pybind"].append(name)
                exports["pybind_details"][name] = {
                    "target": target,
                    "arity": function_arity(text, target),
                    "file": rel,
                }
            for name, target in FFI_EXPORT_RE.findall(source_without_comments):
                exports["tvm_ffi"].append(name)
                exports["tvm_ffi_details"][name] = {
                    "target": target,
                    "arity": function_arity(text, target),
                    "file": rel,
                }
            for category, pattern in CATEGORY_PATTERNS.items():
                lines = line_matches(source_without_comments, pattern)
                if lines:
                    category_counts[category] += len(lines)
                    findings[category].append({"file": rel, "lines": lines})
        elif path.suffix in PYTHON_SUFFIXES:
            result = analyze_python(path, text)
            if any(
                result.get(key)
                for key in (
                    "native_aliases",
                    "keyword_calls",
                    "public_functions",
                    "load_module_lines",
                    "global_registration_lines",
                    "syntax_error",
                )
            ):
                python[rel] = result

    exports["pybind"] = sorted(set(exports["pybind"]))
    exports["tvm_ffi"] = sorted(set(exports["tvm_ffi"]))
    build = detect_build(root, texts)
    risk = risk_assessment(category_counts, build, exports)
    pyi_files = [str(path.relative_to(root)) for path in root.rglob("*.pyi") if path.is_file()]

    return {
        "schema_version": 1,
        "root": str(root),
        "build": build,
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "findings": dict(sorted(findings.items())),
        "exports": exports,
        "python": python,
        "pyi_files": sorted(pyi_files),
        "risk": risk,
    }


def render_markdown(report: dict[str, Any]) -> str:
    build = report["build"]
    lines = [
        f"# Operator Library Audit: `{Path(report['root']).name}`",
        "",
        f"- Root: `{report['root']}`",
        f"- Migration risk: **{report['risk']['level']}** (score {report['risk']['score']})",
        f"- Native exports: {len(report['exports']['pybind'])} pybind11, {len(report['exports']['tvm_ffi'])} tvm-ffi",
        f"- Type stubs: {len(report['pyi_files'])}",
        "",
        "## Build",
        "",
        "| Signal | Value |",
        "| --- | --- |",
    ]
    for key in (
        "has_setup_py",
        "has_pyproject",
        "has_cmake",
        "torch_cpp_extension",
        "scikit_build_core",
        "tvm_ffi_build_dependency",
        "wheel_py_api",
        "find_tvm_ffi",
        "find_torch",
        "python_add_library",
    ):
        lines.append(f"| `{key}` | `{build[key]}` |")
    lines.extend(
        [
            f"| `custom_actions` | `{', '.join(build['custom_actions']) or '-'}` |",
            "",
            "## Coupling Counts",
            "",
            "| Category | Matching lines |",
            "| --- | ---: |",
        ]
    )
    for category, count in report["category_counts"].items():
        lines.append(f"| `{category}` | {count} |")
    lines.extend(["", "## Exports", ""])
    lines.append("- pybind11: " + (", ".join(f"`{x}`" for x in report["exports"]["pybind"]) or "none"))
    lines.append("- tvm-ffi: " + (", ".join(f"`{x}`" for x in report["exports"]["tvm_ffi"]) or "none"))
    details = {**report["exports"]["pybind_details"], **report["exports"]["tvm_ffi_details"]}
    if details:
        lines.extend(["", "| Export | Target | Arity | File |", "| --- | --- | ---: | --- |"])
        for name in sorted(details):
            detail = details[name]
            arity = "?" if detail["arity"] is None else detail["arity"]
            lines.append(f"| `{name}` | `{detail['target']}` | {arity} | `{detail['file']}` |")
    lines.extend(["", "## Highest-Impact Findings", ""])
    if report["risk"]["reasons"]:
        lines.extend(f"- {reason}" for reason in report["risk"]["reasons"])
    else:
        lines.append("- No high-impact patterns detected by the static audit.")
    lines.extend(["", "## Files By Category", ""])
    for category, entries in report["findings"].items():
        lines.append(f"### `{category}`")
        lines.append("")
        for entry in entries:
            preview = ", ".join(str(line) for line in entry["lines"][:12])
            if len(entry["lines"]) > 12:
                preview += ", ..."
            lines.append(f"- `{entry['file']}`: {preview}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="repository root to audit")
    parser.add_argument("--json-out", type=Path, help="write the structured report to this path")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args()

    if not args.repo.is_dir():
        parser.error(f"not a directory: {args.repo}")
    report = audit(args.repo)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
