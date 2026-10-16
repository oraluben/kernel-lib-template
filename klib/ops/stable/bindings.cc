#include "torch213/conversion.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <cstdio>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using Tensor = torch::stable::Tensor;
using Library = std::unique_ptr<void, decltype(&dlclose)>;

extern "C" int klib_cuda_add(float* data, int64_t count, void* stream);

class Runtime {
  using From = int32_t (*)(void*, AtenTensorHandle*);
  using To = int32_t (*)(AtenTensorHandle, void*, void**);
  Library torch_library_{nullptr, dlclose};
  Torch213Conversion compat_;
  From official_from_ = nullptr;
  To official_to_ = nullptr;

  [[noreturn]] static void fail(const char* message) {
    if (PyErr_Occurred()) throw nb::python_error();
    throw std::runtime_error(message);
  }
  Tensor from(nb::handle input) const {
    if (!official_from_) return compat_.from(input);
    AtenTensorHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(official_from_(input.ptr(), &handle));
    return Tensor(handle);
  }
  nb::object to(const Tensor& tensor) const {
    if (!official_to_) return compat_.to(tensor);
    void* output = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(official_to_(tensor.get(), nullptr, &output));
    if (!output) fail("Output conversion returned null");
    return nb::steal<nb::object>(static_cast<PyObject*>(output));
  }

 public:
  Runtime() {
    auto torch = nb::module_::import_("torch");
    unsigned major = 0, minor = 0;
    std::sscanf(nb::str(torch.attr("__version__")).c_str(), "%u.%u", &major, &minor);
    if (major != 2 || minor < 13) throw std::runtime_error("klib requires Torch 2.13 or newer Torch 2.x");
    if (minor == 13) {
      compat_.initialize(torch);
    } else {
      std::string path = nb::str(torch.attr("__file__")).c_str();
      path.resize(path.find_last_of('/'));
      path += "/lib/libtorch_cpu.so";
      torch_library_.reset(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL));
      if (!torch_library_) throw std::runtime_error(dlerror());
      official_from_ = reinterpret_cast<From>(dlsym(torch_library_.get(), "torch_tensor_from_pyobject"));
      official_to_ = reinterpret_cast<To>(dlsym(torch_library_.get(), "torch_tensor_to_pyobject"));
      if (!official_from_ || !official_to_)
        throw std::runtime_error("Torch runtime is missing official stable Tensor interop; no backend fallback on 2.14+");
    }
  }
  nb::dict info() const {
    nb::dict result;
    result["route"] = official_from_ ? "official" : "torch213_aoti";
    result["build_torch"] = KLIB_BUILD_TORCH;
    result["target_torch"] = "2.13";
    result["backend_loaded"] = false;
    result["binding"] = "nanobind_linked_abi3";
    return result;
  }
  nb::object roundtrip(nb::handle input) const { return to(from(input)); }
  nb::object sum(nb::handle input) const { return to(torch::stable::sum(from(input))); }
  nb::object empty_like(nb::handle input) const {
    auto tensor = from(input);
    if (tensor.layout() != torch::headeronly::Layout::Strided)
      throw std::invalid_argument("empty_like requires a strided tensor");
    int32_t dtype, device;
    STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(tensor.get(), &dtype));
    STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(tensor.get(), &device));
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    AtenTensorHandle output = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
        tensor.dim(), sizes.data(), strides.data(), dtype, device, tensor.get_device_index(), &output));
    return to(Tensor(output));
  }
  void add(nb::handle input) {
    auto tensor = from(input);
    if (!tensor.is_cuda() || tensor.layout() != torch::headeronly::Layout::Strided || !tensor.is_contiguous() ||
        tensor.scalar_type() != torch::headeronly::ScalarType::Float)
      throw std::invalid_argument("add requires a contiguous float32 CUDA tensor");
    torch::stable::accelerator::DeviceGuard guard(tensor.get_device_index());
    auto stream = torch::stable::accelerator::getCurrentStream(tensor.get_device_index());
    int status = klib_cuda_add(tensor.mutable_data_ptr<float>(), tensor.numel(), stream.nativeHandle());
    if (status) throw std::runtime_error("CUDA kernel launch failed: " + std::to_string(status));
  }
};

NB_MODULE(_stable, m) {
  nb::class_<Runtime>(m, "Runtime")
    .def(nb::init<>())
    .def("info", &Runtime::info)
    .def("roundtrip", &Runtime::roundtrip)
    .def("sum", &Runtime::sum)
    .def("empty_like", &Runtime::empty_like)
    .def("add", &Runtime::add);
}
