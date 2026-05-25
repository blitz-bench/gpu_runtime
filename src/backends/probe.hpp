#pragma once

#include <utility>
#include <vector>

#include "gpgpu/backend.hpp"
#include "gpgpu/device.hpp"

namespace gpgpu::backends {

// Result of a single backend probe: the Backend descriptor (always populated,
// even when status != Available) and the set of devices it exposed.
struct ProbeResult {
    Backend backend;
    std::vector<Device> devices;
};

ProbeResult probe_cuda();
ProbeResult probe_rocm();
ProbeResult probe_oneapi();
ProbeResult probe_opencl();
ProbeResult probe_vulkan();
ProbeResult probe_metal();

// Convenience helper for assembling a Backend value when the library never
// loaded. The path argument carries the loader-error diagnostic.
Backend make_missing(BackendId id, std::string error);

} // namespace gpgpu::backends
