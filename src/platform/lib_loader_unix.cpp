#if !defined(_WIN32)

#include "lib_loader.hpp"

#include <dlfcn.h>

#include <utility>

namespace gpgpu::platform {

LibHandle::LibHandle(void* h, std::string path) noexcept
    : handle_(h), path_(std::move(path)) {}

LibHandle::LibHandle(LibHandle&& other) noexcept
    : handle_(other.handle_), path_(std::move(other.path_)) {
    other.handle_ = nullptr;
}

LibHandle& LibHandle::operator=(LibHandle&& other) noexcept {
    if (this != &other) {
        reset();
        handle_ = other.handle_;
        path_ = std::move(other.path_);
        other.handle_ = nullptr;
    }
    return *this;
}

LibHandle::~LibHandle() { reset(); }

void LibHandle::reset() noexcept {
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
    path_.clear();
}

void* LibHandle::resolve(const char* symbol) const noexcept {
    if (!handle_ || !symbol) return nullptr;
    dlerror(); // clear
    return dlsym(handle_, symbol);
}

LibHandle try_load(const std::vector<std::string>& candidates,
                   std::string& error_out) {
    std::string aggregate;
    for (const auto& candidate : candidates) {
        if (candidate.empty()) continue;
        dlerror();
        void* h = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (h) {
            return LibHandle(h, candidate);
        }
        const char* err = dlerror();
        if (!aggregate.empty()) aggregate += "; ";
        aggregate += candidate;
        aggregate += ": ";
        aggregate += (err ? err : "unknown error");
    }
    error_out = aggregate.empty() ? std::string("no candidate paths supplied")
                                  : std::move(aggregate);
    return LibHandle();
}

} // namespace gpgpu::platform

#endif // !_WIN32
