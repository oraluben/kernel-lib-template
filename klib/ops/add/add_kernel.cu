#include <cuda.h>

#include <tvm/ffi/container/tensor.h>

#include "tvm_ffi_utils.h"

void add_kernel(const tvm::ffi::TensorView& input) {
    // too lazy to write impl
}


TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, add_kernel);
