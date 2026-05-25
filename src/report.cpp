#include "gpgpu/runtime.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace gpgpu {

std::string_view to_string(Preference p) noexcept {
    switch (p) {
        case Preference::Native:   return "Native";
        case Preference::Fallback: return "Fallback";
        case Preference::Other:    return "Other";
    }
    return "Other";
}

Report::Report(std::vector<Setup> setups, std::vector<Backend> backends)
    : setups_(std::move(setups)), backends_(std::move(backends)) {}

std::vector<Device> Report::devices() const {
    std::vector<Device> out;
    std::unordered_set<std::string> seen;
    out.reserve(setups_.size());
    for (const auto& s : setups_) {
        if (seen.insert(s.device.id()).second) {
            out.push_back(s.device);
        }
    }
    return out;
}

std::vector<Setup> Report::setups_for(std::string_view device_id) const {
    std::vector<Setup> out;
    for (const auto& s : setups_) {
        if (s.device.id() == device_id) out.push_back(s);
    }
    return out;
}

std::vector<Setup> Report::preferred() const {
    std::vector<Setup> out;
    std::unordered_set<std::string> seen;
    // Pass 1: Native picks (one per device id).
    for (const auto& s : setups_) {
        if (s.preferred == Preference::Native && seen.insert(s.device.id()).second) {
            out.push_back(s);
        }
    }
    // Pass 2: Fallback picks for devices that had no Native winner.
    for (const auto& s : setups_) {
        if (s.preferred == Preference::Fallback && seen.insert(s.device.id()).second) {
            out.push_back(s);
        }
    }
    return out;
}

std::vector<Setup> Report::by_backend(BackendId id) const {
    std::vector<Setup> out;
    for (const auto& s : setups_) {
        if (s.backend.id() == id) out.push_back(s);
    }
    return out;
}

std::optional<Backend> Report::backend(BackendId id) const {
    auto it = std::find_if(backends_.begin(), backends_.end(),
                           [id](const Backend& b) { return b.id() == id; });
    if (it == backends_.end()) return std::nullopt;
    return *it;
}

} // namespace gpgpu
