#pragma once

#include <span>

#include <gpgpu/setup.hpp>

#include "../result.hpp"

namespace example {

// Run c = a + b on the ROCm/HIP setup. Caller dispatches only when
// setup.backend.id() == BackendId::ROCm.
RunResult run_vector_add_rocm(const gpgpu::Setup& setup,
                              std::span<const float> a,
                              std::span<const float> b,
                              std::span<float>       c);

} // namespace example
