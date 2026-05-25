#pragma once

#include <cstdint>

namespace gpgpu {

struct Version {
    std::uint16_t major;
    std::uint16_t minor;
    std::uint16_t patch;
};

Version version() noexcept;

} // namespace gpgpu
