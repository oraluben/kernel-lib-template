#include <cuda_runtime.h>
#include "../add/add_kernel.cu"

extern "C" int klib_cuda_add(float* data, int64_t count, void* stream) {
  if (count == 0) return 0;
  AddOneKernel<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(data, count);
  return static_cast<int>(cudaGetLastError());
}
