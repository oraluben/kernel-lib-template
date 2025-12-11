from functools import lru_cache
from pathlib import Path

import tvm_ffi


LIB_ROOT = Path(__file__).parent.parent / "build"
if not LIB_ROOT.exists():
    LIB_ROOT = Path(__file__).parent / "ops"


@lru_cache(maxsize=0)
def load_ffi_lib(name: str):
    """
    libraries would be in `<repo>/build` or `<site-packages>/klib/ops`.

    """
    p = Path(name)

    return tvm_ffi.load_module(LIB_ROOT / p.name)


def add(i):
    load_ffi_lib("add_ffi.so").add(i)


def alloc(t):
    return load_ffi_lib("alloc_ffi.so").empty_like(t)
