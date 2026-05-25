#include "gpgpu/runtime.hpp"

#include <unordered_map>
#include <utility>

#include "backends/probe.hpp"
#include "preference.hpp"

namespace gpgpu {

Report Runtime::query() {
    std::vector<backends::ProbeResult> probes;
    probes.reserve(6);
    probes.push_back(backends::probe_cuda());
    probes.push_back(backends::probe_rocm());
    probes.push_back(backends::probe_oneapi());
    probes.push_back(backends::probe_opencl());
    probes.push_back(backends::probe_vulkan());
    probes.push_back(backends::probe_metal());

    // Group all backend ids per physical device (by Device::id()) so we can
    // assign Preference correctly.
    std::unordered_map<std::string, std::vector<BackendId>> per_device;
    for (const auto& p : probes) {
        for (const auto& dev : p.devices) {
            per_device[dev.id()].push_back(p.backend.id());
        }
    }

    std::vector<Setup> setups;
    std::vector<Backend> backends_out;
    backends_out.reserve(probes.size());

    for (auto& p : probes) {
        backends_out.push_back(p.backend);
        for (auto& dev : p.devices) {
            Setup s;
            s.device = std::move(dev);
            s.backend = p.backend;
            s.preferred = pref::classify(s.device.vendor(), p.backend.id(),
                                         per_device[s.device.id()]);
            setups.push_back(std::move(s));
        }
    }

    return Report(std::move(setups), std::move(backends_out));
}

} // namespace gpgpu
