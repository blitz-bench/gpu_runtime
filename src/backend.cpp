#include "gpgpu/backend.hpp"

#include <utility>

namespace gpgpu {

Backend::Backend(BackendId id,
                 std::string version,
                 std::string path,
                 BackendStatus status,
                 std::optional<std::string> last_error)
    : id_(id),
      version_(std::move(version)),
      path_(std::move(path)),
      status_(status),
      last_error_(std::move(last_error)) {}

std::string_view to_string(BackendId id) noexcept {
    switch (id) {
        case BackendId::CUDA:   return "CUDA";
        case BackendId::ROCm:   return "ROCm";
        case BackendId::OneAPI: return "OneAPI";
        case BackendId::OpenCL: return "OpenCL";
        case BackendId::Vulkan: return "Vulkan";
        case BackendId::Metal:  return "Metal";
    }
    return "Unknown";
}

std::string_view to_string(BackendStatus status) noexcept {
    switch (status) {
        case BackendStatus::Available:     return "Available";
        case BackendStatus::MissingLib:    return "MissingLib";
        case BackendStatus::MissingDriver: return "MissingDriver";
        case BackendStatus::NoDevices:     return "NoDevices";
        case BackendStatus::InitFailed:    return "InitFailed";
    }
    return "Unknown";
}

} // namespace gpgpu
