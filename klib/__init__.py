from importlib.util import find_spec as _find_spec

if _find_spec(__name__ + "._stable") is not None:
    from ._stable_impl import add, empty_like, interop_info, roundtrip, sum

    __all__ = ["add", "empty_like", "interop_info", "roundtrip", "sum"]
else:
    from ._tvm_impl import add, empty_like

    __all__ = ["add", "empty_like"]
