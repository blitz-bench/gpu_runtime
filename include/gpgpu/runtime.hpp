#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include "backend.hpp"
#include "device.hpp"
#include "setup.hpp"

namespace gpgpu {

class Report {
public:
    Report() = default;
    explicit Report(std::vector<Setup> setups, std::vector<Backend> backends);

    using const_iterator = std::vector<Setup>::const_iterator;
    [[nodiscard]] const_iterator begin() const noexcept { return setups_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return setups_.end(); }
    [[nodiscard]] std::size_t size() const noexcept { return setups_.size(); }
    [[nodiscard]] bool empty() const noexcept { return setups_.empty(); }
    [[nodiscard]] const Setup& operator[](std::size_t i) const { return setups_[i]; }
    [[nodiscard]] const std::vector<Setup>& setups() const noexcept { return setups_; }

    // Unique physical devices by Device::id().
    [[nodiscard]] std::vector<Device> devices() const;
    // All Setups that share the given Device::id().
    [[nodiscard]] std::vector<Setup> setups_for(std::string_view device_id) const;
    // Exactly one preferred Setup per device (Native if available, else Fallback).
    [[nodiscard]] std::vector<Setup> preferred() const;
    // All Setups exposed by a particular backend.
    [[nodiscard]] std::vector<Setup> by_backend(BackendId id) const;
    // Includes backends with zero devices so callers can inspect status().
    [[nodiscard]] const std::vector<Backend>& backends() const noexcept { return backends_; }
    [[nodiscard]] std::optional<Backend> backend(BackendId id) const;

private:
    std::vector<Setup> setups_;
    std::vector<Backend> backends_;
};

class Runtime {
public:
    // Synchronous discovery. Probes every backend by dlopen'ing its runtime
    // library from standard install paths, enumerates devices, and returns a
    // Report. Results are not cached - call again to re-probe.
    static Report query();
};

} // namespace gpgpu
