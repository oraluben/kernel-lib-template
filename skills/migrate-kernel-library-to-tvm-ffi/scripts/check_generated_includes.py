#!/usr/bin/env python3
"""Check include paths embedded in runtime-generated C/CUDA source."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


INCLUDE_RE = re.compile(r"#\s*include\s*[<\"]([^>\"]+)[>\"]")


def parse_expansion(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expansion must look like 'sm{}=90,100'")
    pattern, raw_values = value.split("=", 1)
    values = [item for item in raw_values.split(",") if item]
    if "{}" not in pattern or not values:
        raise argparse.ArgumentTypeError("expansion must contain '{}' and at least one value")
    return pattern, values


def expand(path: str, expansions: list[tuple[str, list[str]]]) -> list[str]:
    results = [path]
    for pattern, values in expansions:
        next_results: list[str] = []
        for item in results:
            if pattern in item:
                next_results.extend(item.replace(pattern, pattern.replace("{}", value)) for value in values)
            else:
                next_results.append(item)
        results = next_results
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="file or directory containing generated-source builders")
    parser.add_argument("--include-root", type=Path, action="append", required=True)
    parser.add_argument("--expand", type=parse_expansion, action="append", default=[])
    args = parser.parse_args()

    files = [args.source] if args.source.is_file() else sorted(
        path for path in args.source.rglob("*") if path.is_file() and path.suffix in {".h", ".hh", ".hpp", ".cc", ".cpp"}
    )
    roots = [root.resolve() for root in args.include_root]
    missing: list[tuple[Path, str]] = []
    checked: set[str] = set()

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw in INCLUDE_RE.findall(text):
            for include in expand(raw, args.expand):
                if include in checked:
                    continue
                checked.add(include)
                if not any((root / include).exists() for root in roots):
                    missing.append((path, include))

    print(f"checked {len(checked)} unique include paths across {len(files)} files")
    if missing:
        for source, include in missing:
            print(f"MISSING {include} (referenced by {source})")
        return 1
    print("all generated-source include paths resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
