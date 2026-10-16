#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void AddOneKernel(float* x, int64_t n) {
    int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += 1;
}

extern "C" int klib_cuda_add(float* data, int64_t count, void* stream) {
    if (count == 0) return 0;
    AddOneKernel<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(data, count);
    return static_cast<int>(cudaGetLastError());
}
