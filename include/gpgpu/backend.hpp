#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace gpgpu {

enum class BackendId : std::uint8_t {
    CUDA,
    ROCm,
    OneAPI,
    OpenCL,
    Vulkan,
    Metal,
};

enum class BackendStatus : std::uint8_t {
    Available,
    MissingLib,
    MissingDriver,
    NoDevices,
    InitFailed,
};

std::string_view to_string(BackendId id) noexcept;
std::string_view to_string(BackendStatus status) noexcept;

class Backend {
public:
    Backend() = default;
    Backend(BackendId id,
            std::string version,
            std::string path,
            BackendStatus status,
            std::optional<std::string> last_error);

    [[nodiscard]] BackendId id() const noexcept { return id_; }
    [[nodiscard]] std::string_view name() const noexcept { return to_string(id_); }
    [[nodiscard]] const std::string& version() const noexcept { return version_; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] BackendStatus status() const noexcept { return status_; }
    [[nodiscard]] const std::optional<std::string>& last_error() const noexcept { return last_error_; }

private:
    BackendId id_{BackendId::OpenCL};
    std::string version_;
    std::string path_;
    BackendStatus status_{BackendStatus::MissingLib};
    std::optional<std::string> last_error_;
};

} // namespace gpgpu
