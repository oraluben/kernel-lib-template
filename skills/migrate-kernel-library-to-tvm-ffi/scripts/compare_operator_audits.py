#!/usr/bin/env python3
"""Compare two JSON reports produced by audit_operator_library.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def delta(before: dict[str, int], after: dict[str, int], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    before = load(args.before)
    after = load(args.after)
    old_exports = set(before["exports"]["pybind"]) | set(before["exports"]["tvm_ffi"])
    new_exports = set(after["exports"]["pybind"]) | set(after["exports"]["tvm_ffi"])
    removed = sorted(old_exports - new_exports)
    added = sorted(new_exports - old_exports)
    before_details = {
        **before["exports"].get("pybind_details", {}),
        **before["exports"].get("tvm_ffi_details", {}),
    }
    after_details = {
        **after["exports"].get("pybind_details", {}),
        **after["exports"].get("tvm_ffi_details", {}),
    }
    arity_changes = []
    for name in sorted(old_exports & new_exports):
        old_arity = before_details.get(name, {}).get("arity")
        new_arity = after_details.get(name, {}).get("arity")
        if old_arity is not None and new_arity is not None and old_arity != new_arity:
            arity_changes.append((name, old_arity, new_arity))

    print(f"# Operator Audit Comparison\n")
    print(f"- Before: `{before['root']}`")
    print(f"- After: `{after['root']}`")
    print(f"- Risk: **{before['risk']['level']} ({before['risk']['score']})** -> "
          f"**{after['risk']['level']} ({after['risk']['score']})**")
    print("\n## Export Surface\n")
    print("- Removed: " + (", ".join(f"`{name}`" for name in removed) or "none"))
    print("- Added: " + (", ".join(f"`{name}`" for name in added) or "none"))
    print("- Arity changes: " + (
        ", ".join(f"`{name}` {old}->{new}" for name, old, new in arity_changes) or "none"
    ))
    print("\n## Coupling Delta\n")
    print("| Category | Before | After | Delta |")
    print("| --- | ---: | ---: | ---: |")
    keys = sorted(set(before["category_counts"]) | set(after["category_counts"]))
    for key in keys:
        old = int(before["category_counts"].get(key, 0))
        new = int(after["category_counts"].get(key, 0))
        print(f"| `{key}` | {old} | {new} | {new - old:+d} |")

    failures: list[str] = []
    if removed:
        failures.append("native exports were removed")
    if arity_changes:
        failures.append("native export arity changed")
    if after["category_counts"].get("binding", 0) and after["exports"]["pybind"]:
        failures.append("pybind11 exports remain")
    if after["build"].get("torch_cpp_extension"):
        failures.append("torch.utils.cpp_extension remains the build backend")
    if not after["exports"]["tvm_ffi"]:
        failures.append("no tvm-ffi exports detected")

    print("\n## Automated Verdict\n")
    if failures:
        for failure in failures:
            print(f"- FAIL: {failure}")
        return 1
    print("- PASS: export names are preserved and the expected tvm-ffi/build signals are present.")
    print("- This verdict is static only; compile, wheel, ABI, GPU correctness, and performance still require validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
