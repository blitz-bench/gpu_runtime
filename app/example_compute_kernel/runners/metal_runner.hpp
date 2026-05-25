#pragma once

#include <span>

#include <gpgpu/setup.hpp>

#include "../result.hpp"

namespace example {

// Run c = a + b on the Metal setup. Caller dispatches only when
// setup.backend.id() == BackendId::Metal. The implementation lives in
// metal_runner.mm and only links on Apple; on other platforms this header
// is still included by main.cpp and a stub provides the symbol.
RunResult run_vector_add_metal(const gpgpu::Setup& setup,
                               std::span<const float> a,
                               std::span<const float> b,
                               std::span<float>       c);

} // namespace example
