#include <cuda.h>

extern "C" __global__ void AddOneKernel(float* x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] += 1;
}
