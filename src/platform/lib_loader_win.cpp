#if defined(_WIN32)

#include "lib_loader.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <utility>

namespace gpgpu::platform {

namespace {
std::string last_error_string() {
    DWORD code = ::GetLastError();
    if (code == 0) return {};
    LPSTR buf = nullptr;
    DWORD len = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buf), 0, nullptr);
    std::string out;
    if (len && buf) {
        out.assign(buf, len);
        while (!out.empty() && (out.back() == '\r' || out.back() == '\n' || out.back() == ' ')) {
            out.pop_back();
        }
    }
    if (buf) ::LocalFree(buf);
    if (out.empty()) {
        out = "GetLastError=" + std::to_string(code);
    }
    return out;
}
} // namespace

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
        ::FreeLibrary(reinterpret_cast<HMODULE>(handle_));
        handle_ = nullptr;
    }
    path_.clear();
}

void* LibHandle::resolve(const char* symbol) const noexcept {
    if (!handle_ || !symbol) return nullptr;
    return reinterpret_cast<void*>(
        ::GetProcAddress(reinterpret_cast<HMODULE>(handle_), symbol));
}

namespace {
bool is_absolute_path(const std::string& p) {
    if (p.size() >= 3 && (p[1] == ':') && (p[2] == '\\' || p[2] == '/')) return true;
    if (!p.empty() && (p[0] == '\\' || p[0] == '/')) return true;
    return false;
}
} // namespace

LibHandle try_load(const std::vector<std::string>& candidates,
                   std::string& error_out) {
    std::string aggregate;
    for (const auto& candidate : candidates) {
        if (candidate.empty()) continue;
        // LOAD_WITH_ALTERED_SEARCH_PATH requires an absolute path; for bare
        // filenames we let the OS loader follow its default search order
        // (which includes System32, where OpenCL.dll / vulkan-1.dll live).
        DWORD flags = is_absolute_path(candidate) ? LOAD_WITH_ALTERED_SEARCH_PATH : 0;
        HMODULE h = ::LoadLibraryExA(candidate.c_str(), nullptr, flags);
        if (h) {
            return LibHandle(reinterpret_cast<void*>(h), candidate);
        }
        std::string err = last_error_string();
        if (!aggregate.empty()) aggregate += "; ";
        aggregate += candidate;
        aggregate += ": ";
        aggregate += err;
    }
    error_out = aggregate.empty() ? std::string("no candidate paths supplied")
                                  : std::move(aggregate);
    return LibHandle();
}

} // namespace gpgpu::platform

#endif // _WIN32
