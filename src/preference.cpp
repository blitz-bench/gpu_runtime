#include "preference.hpp"

#include <algorithm>
#include <optional>

namespace gpgpu::pref {

namespace {

std::optional<BackendId> native_backend(Vendor v) {
    switch (v) {
        case Vendor::Nvidia: return BackendId::CUDA;
        case Vendor::AMD:    return BackendId::ROCm;
        case Vendor::Intel:  return BackendId::OneAPI;
        case Vendor::Apple:  return BackendId::Metal;
        default:             return std::nullopt;
    }
}

bool contains(const std::vector<BackendId>& v, BackendId id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

BackendId pick_fallback(const std::vector<BackendId>& available) {
    if (contains(available, BackendId::OpenCL)) return BackendId::OpenCL;
    if (contains(available, BackendId::Vulkan)) return BackendId::Vulkan;
    // Otherwise the first available wins (e.g. only Metal on macOS for a
    // non-Apple GPU — unlikely but defined).
    return available.empty() ? BackendId::OpenCL : available.front();
}

} // namespace

Preference classify(Vendor vendor,
                    BackendId backend,
                    const std::vector<BackendId>& available_for_device) {
    if (auto native = native_backend(vendor); native && contains(available_for_device, *native)) {
        return backend == *native ? Preference::Native : Preference::Other;
    }
    const BackendId winner = pick_fallback(available_for_device);
    return backend == winner ? Preference::Fallback : Preference::Other;
}

} // namespace gpgpu::pref
