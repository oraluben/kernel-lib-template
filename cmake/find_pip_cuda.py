"""Locate pip-installed CUDA toolkit and prepare it for CMake consumption.

Used by cmake/FindPipCUDAToolkit.cmake via ``execute_process``.
Outputs a JSON object with paths on success, exits with code 1 on failure.
"""

import importlib.util
import glob
import json
import os
import subprocess
import sys
import tempfile


def main():
    result = {}
    # nvidia-cuda-nvcc (v13+) installs nvcc under nvidia/cu<ver>/bin/
    spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    if spec is None:
        sys.exit(1)
    nvidia_dir = os.path.dirname(spec.submodule_search_locations[0])
    # Find the cu* directory (e.g., cu13)
    cu_dirs = [
        d for d in os.listdir(nvidia_dir) if d.startswith("cu") and d[2:].isdigit()
    ]
    if not cu_dirs:
        sys.exit(1)
    cu_dir = os.path.join(nvidia_dir, sorted(cu_dirs)[-1])
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
    # pip packages use lib/ but nvcc expects lib64/ on 64-bit; create symlink if needed
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
    # Create a minimal stub so that -lcuda linking succeeds at build time;
    # at runtime the real driver is loaded via dlopen/RPATH.
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
