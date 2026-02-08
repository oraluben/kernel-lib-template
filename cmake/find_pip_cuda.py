"""Locate pip-installed CUDA toolkit and prepare it for CMake consumption.

Used by cmake/FindPipCUDAToolkit.cmake via ``execute_process``.
Outputs a JSON object with paths on success, exits with code 1 on failure.
"""

import glob
import json
import os
import site
import subprocess
import sys
import tempfile


def _find_nvidia_cu_dir():
    """Search all site-packages directories for nvidia/cu*/bin/nvcc.

    This works even when running inside pip's isolated build environment
    because we scan the actual site-packages paths rather than relying on
    importlib (which only sees the isolated env).
    """
    search_dirs = site.getsitepackages() + [site.getusersitepackages()]
    for sp in search_dirs:
        nvidia_dir = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        cu_dirs = [
            d
            for d in os.listdir(nvidia_dir)
            if d.startswith("cu") and d[2:].isdigit()
        ]
        if not cu_dirs:
            continue
        # Use the highest versioned directory (e.g., cu13 over cu9)
        cu_dir = os.path.join(
            nvidia_dir, max(cu_dirs, key=lambda d: int(d[2:]))
        )
        nvcc = os.path.join(cu_dir, "bin", "nvcc")
        if os.path.isfile(nvcc):
            return cu_dir
    return None


def main():
    result = {}
    # nvidia-cuda-nvcc (v13+) installs nvcc under nvidia/cu<ver>/bin/
    cu_dir = _find_nvidia_cu_dir()
    if cu_dir is None:
        sys.exit(1)
    nvcc = os.path.join(cu_dir, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        sys.exit(1)
    result["nvcc"] = nvcc
    result["root"] = cu_dir
    result["include"] = os.path.join(cu_dir, "include")

    # Check for CCCL headers
    cccl_include = os.path.join(cu_dir, "include", "cccl")
    if os.path.isdir(cccl_include):
        result["cccl_include"] = cccl_include

    lib_dir = os.path.join(cu_dir, "lib")
    # pip packages use lib/ but nvcc expects lib64/ on 64-bit; create a relative
    # symlink so paths work regardless of the absolute install location.
    lib64_dir = os.path.join(cu_dir, "lib64")
    if os.path.isdir(lib_dir) and not os.path.exists(lib64_dir):
        try:
            os.symlink("lib", lib64_dir)
        except OSError:
            pass

    # pip packages may ship versioned .so (e.g., libcudart.so.13) without the
    # unversioned symlink that CMake's FindCUDAToolkit expects (libcudart.so).
    if os.path.isdir(lib_dir):
        for versioned in glob.glob(os.path.join(lib_dir, "*.so.*")):
            base = versioned.split(".so.")[0] + ".so"
            if not os.path.exists(base):
                try:
                    os.symlink(os.path.basename(versioned), base)
                except OSError:
                    pass

    # pip packages don't include the CUDA driver stub (libcuda.so).
    # Create a minimal stub so that -lcuda linking succeeds at build time.
    # Only the .so file needs to exist; the actual symbol is unused because
    # the real libcuda.so from the GPU driver is loaded at runtime.
    stubs_dir = os.path.join(lib_dir, "stubs")
    stub_path = os.path.join(stubs_dir, "libcuda.so")
    if not os.path.exists(stub_path):
        os.makedirs(stubs_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write("void cuGetErrorString(void){}\n")
            f.flush()
            try:
                subprocess.check_call(
                    ["gcc", "-shared", "-o", stub_path, f.name],
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            finally:
                os.unlink(f.name)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
