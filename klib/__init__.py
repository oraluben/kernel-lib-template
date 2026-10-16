from importlib.util import find_spec
from pathlib import Path

if find_spec(__name__ + "._stable") is None:
    from ._tvm_impl import add, empty_like
else:
    import torch

    from ._stable import Runtime

    _runtime = Runtime(str(Path(__file__).parent / "_cuda_add.so"))

    def add(tensor):
        """Increment a contiguous float32 CPU/CUDA tensor in place, without autograd."""
        if tensor.requires_grad:
            raise RuntimeError("klib.add is an inference-only kernel")
        _runtime.add(tensor)
        torch.autograd.graph.increment_version(tensor)

    def empty_like(tensor):
        """Allocate a tensor preserving shape, strides, dtype and device."""
        return _runtime.empty_like(tensor)

    def sum(tensor):
        return _runtime.sum(tensor)

    def roundtrip(tensor):
        return _runtime.roundtrip(tensor)

    def interop_info():
        return _runtime.info()
