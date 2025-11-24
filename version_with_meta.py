from __future__ import annotations

import os

__all__ = ["dynamic_metadata"]

base_version = "0.1.0"


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
) -> str:
    try:
        assert field == "version"

        patched_version = base_version

        # VERSION_SUFFIX='+cu128'
        if version_ext := os.environ.get("VERSION_SUFFIX"):
            patched_version += version_ext
        elif cuda_version := os.environ.get("CUDA_VERSION"):
            major, minor, *_ = cuda_version.split(".")
            backend = f"+cu{major}{minor}"
            patched_version += backend

        return patched_version
    except:
        return base_version
