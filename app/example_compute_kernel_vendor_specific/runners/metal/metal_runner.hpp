#pragma once

#include <span>

#include <gpgpu/setup.hpp>

#include "../../result.hpp"

namespace example {

// Run c = a + b on a Metal setup. Built only when its SDK / framework was
// detected at configure time; the lib target sets EXAMPLE_VS_HAVE_METAL
// which gates inclusion from main.cpp.
RunResult run_vector_add_metal(const gpgpu::Setup& setup,
                              std::span<const float> a,
                              std::span<const float> b,
                              std::span<float>       c);

} // namespace example
