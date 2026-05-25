#pragma once

#include <string>

#include <gpgpu/backend.hpp>
#include <gpgpu/setup.hpp>

#include "timings.hpp"

// The ONLY type shared across runners. Pure data; carries no behaviour.
// Each runner constructs and returns one per invocation.
namespace example {

struct RunResult {
    std::string       device_id;     // gpgpu::Device::id()
    std::string       device_name;   // human readable
    gpgpu::BackendId  backend{gpgpu::BackendId::OpenCL};
    gpgpu::Preference preferred{gpgpu::Preference::Other};
    Timings           timings{};
    bool              correct{false};
    std::string       error;         // empty on success
};

} // namespace example
