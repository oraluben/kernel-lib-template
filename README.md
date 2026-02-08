# Example

## Quick Start

### With pip-provided CUDA toolchain (no host CUDA required)

```bash
pip install nvidia-cuda-nvcc nvidia-cuda-cccl
pip install . -v
```

### With host CUDA toolchain

```bash
# Ensure CUDA toolkit is installed on the host (e.g., /usr/local/cuda)
pip install . -v
```

### Run tests

```
python tests/*.py
```
