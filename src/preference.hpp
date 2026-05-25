#pragma once

#include <vector>

#include "gpgpu/backend.hpp"
#include "gpgpu/device.hpp"
#include "gpgpu/setup.hpp"

namespace gpgpu::pref {

// Pick the Preference of a single (vendor, backend) pair, given the set of
// backends that exposed this physical device.
//
//   - The vendor-native backend (NVIDIA->CUDA, AMD->ROCm, Intel->OneAPI,
//     Apple->Metal) is Preference::Native when present.
//   - If no native backend is available for the vendor, OpenCL wins as
//     Preference::Fallback; otherwise Vulkan as Fallback; otherwise the only
//     remaining backend is the Fallback.
//   - All other Setups for the same device are Preference::Other.
Preference classify(Vendor vendor,
                    BackendId backend,
                    const std::vector<BackendId>& available_for_device);

} // namespace gpgpu::pref
