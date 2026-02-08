# Example

## Quick Start

### With host CUDA toolchain

```bash
# Ensure CUDA toolkit is installed on the host (e.g., /usr/local/cuda)
pip install . -v
```

### With pip-provided CUDA toolchain (no host CUDA required)

Option A — pip toolchain in the current environment (use `--no-build-isolation`):

```bash
pip install nvidia-cuda-nvcc nvidia-cuda-cccl
pip install . -v --no-build-isolation
```

Option B — pip toolchain in another virtualenv or path:

```bash
# Point to the cu<ver> directory inside another venv's site-packages
export WITH_PIP_CUDA_TOOLCHAIN=/path/to/venv/lib/python3.x/site-packages/nvidia/cu13
pip install . -v
```

### Run tests

```
python tests/*.py
```
