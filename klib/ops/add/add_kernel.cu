#include <cuda.h>
#include <stdint.h>

extern "C" __global__ void AddOneKernel(float* x, int64_t n) {
    int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += 1;
}
