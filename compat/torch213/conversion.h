#pragma once
#include <nanobind/nanobind.h>
#include <torch/csrc/stable/tensor.h>

class Torch213Conversion {
  nanobind::object from_, to_;

 public:
  void initialize(nanobind::handle torch) {
    auto aoti = torch.attr("_C").attr("_aoti");
    from_ = aoti.attr("unsafe_alloc_void_ptr_from_tensor");
    to_ = aoti.attr("alloc_tensor_by_stealing_from_void_ptr");
  }
  torch::stable::Tensor from(nanobind::handle input) const {
    auto capsule = from_(input);
    auto handle = static_cast<AtenTensorHandle>(PyCapsule_GetPointer(capsule.ptr(), nullptr));
    if (!handle) throw nanobind::python_error();
    return torch::stable::Tensor(handle);
  }
  nanobind::object to(const torch::stable::Tensor& tensor) const {
    // In Torch 2.13 the singular function copies; despite its name, it does not free the handle.
    return to_(nanobind::capsule(tensor.get()));
  }
};
