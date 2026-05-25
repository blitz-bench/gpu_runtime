#pragma once

#include <span>

#include <gpgpu/setup.hpp>

#include "../result.hpp"

namespace example {

// Run c = a + b on the CUDA setup. Caller is responsible for dispatching
// only when setup.backend.id() == BackendId::CUDA. Returns a populated
// RunResult; sets `error` when initialisation / launch fails.
RunResult run_vector_add_cuda(const gpgpu::Setup& setup,
                              std::span<const float> a,
                              std::span<const float> b,
                              std::span<float>       c);

} // namespace example
