#include <cuda.h>

#include <tvm/ffi/container/tensor.h>
// #include <tvm/ffi/extra/cuda/cubin_launcher.h>

#include "tvm_ffi_utils.h"

#include "add_kernel.h"

void add_kernel(const tvm::ffi::TensorView& input) {
    CUresult res;

    CUmodule module;
    res = cuModuleLoadFatBinary(&module, imageBytes);
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to load fatbin! Error code: " << res << std::endl;
        return;
    }

    CUfunction kernel;
    res = cuModuleGetFunction(&kernel, module, "AddOneKernel");
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to load kernel! Error code: " << res << std::endl;
        return;
    }

    void* input_ptr = input.data_ptr();
    void *args[] = { &input_ptr };

    long N = input.numel();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUstream stream = static_cast<CUstream>(TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    cuLaunchKernel(
        kernel,
        blocksPerGrid, 1, 1,
        threadsPerBlock, 1, 1,
        0,
        stream,
        args,
        NULL
    );
}


TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, add_kernel);
