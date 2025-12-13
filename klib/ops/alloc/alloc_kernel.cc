#include <tvm/ffi/container/tensor.h>

#include <tvm_ffi_utils.h>

tvm::ffi::Tensor empty_like_kernel(const tvm::ffi::TensorView& input) {
    int64_t cosize = 1;
    for (int i = 0; i < input.ndim(); i++) {
        cosize += (input.shape().at(i) - 1) * input.stride(i);
    }
    return tvm::ffi::Tensor::FromEnvAlloc(
        TVMFFIEnvTensorAlloc, {cosize}, input.dtype(), input.device()
    ).as_strided(input.shape(), input.strides());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(empty_like, empty_like_kernel);
