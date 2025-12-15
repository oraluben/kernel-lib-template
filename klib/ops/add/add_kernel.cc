#include <cuda.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>

#include "driver_types.h"
#include "tvm_ffi_utils.h"

#include "add_kernel.h"

#define TVM_FFI_EMBED_CUBIN_v2(name, imageBytes)      \
  namespace {                                  \
  struct EmbedCubinModule_##name {             \
    tvm::ffi::CubinModule mod{imageBytes};     \
    static EmbedCubinModule_##name* Global() { \
      static EmbedCubinModule_##name inst;     \
      return &inst;                            \
    }                                          \
  };                                           \
  } /* anonymous namespace */

TVM_FFI_EMBED_CUBIN_v2(my_cubin, imageBytes);

void add_kernel(const tvm::ffi::TensorView& input) {
    static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_cubin, "AddOneKernel");

    void* input_ptr = input.data_ptr();
    void *args[] = { &input_ptr };

    long N = input.numel();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUstream stream = static_cast<CUstream>(TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    tvm::ffi::dim3 grid((N + threadsPerBlock - 1) / threadsPerBlock);
    tvm::ffi::dim3 block(256);

    cudaError_t result = kernel.Launch(args, grid, block, stream);
    TVM_FFI_CHECK_CUDA_ERROR(result);
}


TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, add_kernel);
