#include <tvm/ffi/container/tensor.h>

#include <tvm_ffi_utils.h>

tvm::ffi::Tensor empty_kernel(const tvm::ffi::TensorView& input) {
    return empty(input.shape(), dl_bfloat16, input.device());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(empty_like, empty_kernel);
