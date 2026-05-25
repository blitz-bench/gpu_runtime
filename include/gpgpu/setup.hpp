#pragma once

#include <cstdint>
#include <string_view>

#include "backend.hpp"
#include "device.hpp"

namespace gpgpu {

enum class Preference : std::uint8_t {
    Native,   // vendor-native backend was picked (NVIDIA->CUDA, AMD->ROCm, ...)
    Fallback, // OpenCL or Vulkan picked because no native backend available
    Other,    // not the preferred Setup for this device
};

std::string_view to_string(Preference p) noexcept;

struct Setup {
    Device device;
    Backend backend;
    Preference preferred{Preference::Other};
};

} // namespace gpgpu
