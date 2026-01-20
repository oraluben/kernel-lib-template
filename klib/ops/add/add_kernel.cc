#include <cuda.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>

#include "tvm_ffi_utils.h"

#include "add_kernel.h"

TVM_FFI_EMBED_CUBIN(env);

void add_kernel(const tvm::ffi::TensorView &input) {
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(env, "AddOneKernel");

  int64_t n = input.numel();
  tvm::ffi::dim3 grid((n + 255) / 256);
  tvm::ffi::dim3 block(256);

  auto stream = static_cast<tvm::ffi::cuda_api::StreamHandle>(
      TVM_FFI_GET_CUDA_STREAM(input));

  void *input_ptr = input.data_ptr();
  void *args[] = {&input_ptr};
  tvm::ffi::cuda_api::ResultType result =
      kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(result);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, add_kernel);
