#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace gpgpu::platform {

// Opaque OS-specific handle returned by dlopen / LoadLibrary.
class LibHandle {
public:
    LibHandle() = default;
    LibHandle(void* h, std::string path) noexcept;
    LibHandle(const LibHandle&) = delete;
    LibHandle& operator=(const LibHandle&) = delete;
    LibHandle(LibHandle&& other) noexcept;
    LibHandle& operator=(LibHandle&& other) noexcept;
    ~LibHandle();

    [[nodiscard]] bool is_open() const noexcept { return handle_ != nullptr; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] void* raw() const noexcept { return handle_; }
    void* resolve(const char* symbol) const noexcept;

    void reset() noexcept;

private:
    void* handle_{nullptr};
    std::string path_;
};

// Try to load the first existing shared library from a list of candidate paths
// (each may be absolute, or a bare filename to resolve via the OS loader's
// default search). Returns an open LibHandle on success, an empty handle and a
// populated `error_out` on failure.
LibHandle try_load(const std::vector<std::string>& candidates,
                   std::string& error_out);

// Look up a symbol and cast it to the requested function pointer type.
template <typename F>
F resolve_as(const LibHandle& lib, const char* symbol) noexcept {
    return reinterpret_cast<F>(lib.resolve(symbol));
}

} // namespace gpgpu::platform
