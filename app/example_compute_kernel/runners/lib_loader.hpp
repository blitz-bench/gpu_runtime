#pragma once

// Tiny header-only OS shim so each runner can re-open the shared library
// gpu_runtime already located (via setup.backend.path()) without having to
// duplicate dlopen / LoadLibrary boilerplate. Intentionally NOT a backend
// abstraction — every runner still drives its backend directly.

#include <string>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace example::loader {

class Lib {
public:
    Lib() = default;
    explicit Lib(const std::string& path) {
#if defined(_WIN32)
        handle_ = ::LoadLibraryA(path.c_str());
#else
        handle_ = ::dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
    }
    Lib(const Lib&) = delete;
    Lib& operator=(const Lib&) = delete;
    Lib(Lib&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Lib& operator=(Lib&& other) noexcept {
        if (this != &other) {
            close();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    ~Lib() { close(); }

    [[nodiscard]] bool ok() const noexcept { return handle_ != nullptr; }

    template <typename Fn>
    Fn sym(const char* name) const noexcept {
        if (!handle_) return nullptr;
#if defined(_WIN32)
        return reinterpret_cast<Fn>(::GetProcAddress(
            reinterpret_cast<HMODULE>(handle_), name));
#else
        return reinterpret_cast<Fn>(::dlsym(handle_, name));
#endif
    }

private:
    void close() noexcept {
        if (handle_) {
#if defined(_WIN32)
            ::FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
            ::dlclose(handle_);
#endif
            handle_ = nullptr;
        }
    }
    void* handle_{nullptr};
};

} // namespace example::loader
