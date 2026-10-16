import gc
import importlib.abc
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import torch

if not torch.cuda.is_available():
    raise RuntimeError("The kernel tests require a CUDA-enabled Torch and a GPU")

BLOCKED = {"nanobind_backend", "_klib_torch213", "tvm_ffi"}


class RejectBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("Blocked backend: " + fullname)


sys.meta_path.insert(0, RejectBackend())
import klib
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests


class TestLoading(TestCase):
    def test_build_and_route(self):
        info = klib.interop_info()
        self.assertGreaterEqual(tuple(map(int, info["build_torch"].split(".")[:2])), (2, 14))
        self.assertEqual(info["target_torch"], "2.13")
        old = torch.__version__.startswith("2.13.")
        self.assertEqual(info["route"], "torch213_aoti" if old else "official")
        self.assertEqual(info["binding"], "nanobind_linked_abi3")
        self.assertFalse(info["backend_loaded"])
        self.assertFalse(BLOCKED.intersection(sys.modules))
        if expected := os.environ.get("KLIB_EXPECTED_EXTENSION_SHA256"):
            self.assertEqual(hashlib.sha256(Path(klib._stable.__file__).read_bytes()).hexdigest(), expected)

    def test_missing_backend_contract(self):
        code = r'''
import importlib.abc
import sys
import torch
names = {"nanobind_backend", "_klib_torch213", "tvm_ffi"}
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in names:
            raise ModuleNotFoundError("Blocked custom backend: " + fullname)
sys.meta_path.insert(0, Reject())
import klib
x = torch.arange(7, dtype=torch.float32)
assert klib.sum(x).item() == 21
assert not any(name in sys.modules for name in names)
print("backend absence contract passed")
'''
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                              cwd=os.path.dirname(os.path.dirname(klib.__file__)))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_add_rejects_cpu(self):
        value = torch.zeros(7)
        with self.assertRaisesRegex(ValueError, "CUDA tensor"):
            klib.add(value)
        self.assertEqual(value, torch.zeros_like(value))

    def test_aoti_only_required_on_213(self):
        code = r'''
import torch
del torch._C._aoti.unsafe_alloc_void_ptr_from_tensor
del torch._C._aoti.alloc_tensor_by_stealing_from_void_ptr
if torch.__version__.startswith("2.13."):
    try:
        import klib
    except AttributeError:
        pass
    else:
        raise AssertionError("2.13 unexpectedly worked without its conversion functions")
else:
    import klib
    assert klib.sum(torch.ones(7)).item() == 7
'''
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                              cwd=os.path.dirname(os.path.dirname(klib.__file__)))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class TestOperators(TestCase):
    @parametrize("size", [0, 1, 255, 256, 257, 1025])
    def test_add_and_bounds(self, device, size):
        storage = torch.full((size + 32,), -7.0, device=device)
        value = storage[16:16 + size]
        value.copy_(torch.arange(size, dtype=torch.float32, device=device))
        expected = storage.clone()
        expected[16:16 + size].add_(1)
        version = value._version
        self.assertIsNone(klib.add(value))
        self.assertEqual(storage, expected)
        self.assertGreater(value._version, version)

    @parametrize("dtype", [torch.float32, torch.int64, torch.bool])
    @parametrize("shape", [(), (0,), (2, 3)])
    def test_roundtrip(self, device, dtype, shape):
        value = torch.ones(shape, dtype=dtype, device=device)
        self.assertIs(klib.roundtrip(value), value)
        self.assertEqual(klib.roundtrip(value), value)

    @parametrize("shape,stride", [([], []), ([0, 3], [3, 1]), ([10, 1], [100, 1]),
                                 ([3, 4], [1, 3]), ([3, 4], [9, 2])])
    def test_empty_like_preserves_strides(self, device, shape, stride):
        value = torch.empty_strided(shape, stride, device=device)
        result = klib.empty_like(value)
        self.assertEqual(result.shape, value.shape)
        self.assertEqual(result.stride(), value.stride())
        self.assertEqual(result.dtype, value.dtype)
        self.assertEqual(result.device, value.device)
        if value.numel():
            self.assertNotEqual(result.data_ptr(), value.data_ptr())
        result.fill_(4)
        self.assertEqual(result, torch.full(shape, 4.0, device=device))

    @parametrize("shape", [(), (0,), (3, 4)])
    def test_stable_sum(self, device, shape):
        value = torch.randn(shape, device=device, requires_grad=True)
        result = klib.sum(value)
        self.assertEqual(result, value.sum())
        result.backward()
        self.assertEqual(value.grad, torch.ones_like(value))

    def test_parameter_view_and_lifetime(self, device):
        value = torch.nn.Parameter(torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4))
        self.assertIs(klib.roundtrip(value), value)
        view = value.t()
        self.assertIs(klib.roundtrip(view), view)
        count = sys.getrefcount(view)
        for _ in range(500):
            klib.roundtrip(view)
        self.assertEqual(sys.getrefcount(view), count)
        result = klib.empty_like(view)
        weak = torch._C._WeakTensorRef(value)
        del value, view
        gc.collect()
        self.assertTrue(weak.expired())
        result.fill_(2)
        self.assertEqual(result, torch.full((4, 3), 2.0, device=device))

    def test_invalid_input_and_recovery(self, device):
        for invalid in (None, "not a tensor", 1):
            with self.assertRaises((RuntimeError, TypeError)):
                klib.roundtrip(invalid)
        value = torch.ones(2, 3, device=device)
        for invalid in (value.t(), value.to(torch.int64), value.clone().requires_grad_()):
            with self.assertRaises((RuntimeError, ValueError)):
                klib.add(invalid)
        self.assertEqual(klib.sum(value), value.sum())

    def test_mutation_version_detected(self, device):
        value = torch.ones(7, device=device, requires_grad=True)
        scale = torch.ones_like(value)
        output = (value * scale).sum()
        klib.add(scale)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
            output.backward()

    def test_nondefault_stream(self, device):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value = torch.ones(257, device=device)
            klib.add(value)
            result = klib.sum(value)
            output = klib.empty_like(value)
            output.copy_(value)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertEqual(result, torch.tensor(514.0, device=device))
        self.assertEqual(output, value)


instantiate_device_type_tests(TestOperators, globals(), only_for="cuda")
if __name__ == "__main__":
    run_tests()
